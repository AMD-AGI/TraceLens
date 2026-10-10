###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Symbolic tensor shape and dtype inference for model computation graphs."""

from __future__ import annotations

import ast
import functools
import inspect
import json
import copy
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import Counter
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from TraceLens.ModelUtils.config_resolve import declared_config_aliases
from TraceLens.ModelUtils.extract import (
    architecture_section_trees,
    find_vision_tower,
    vision_scoped_classes,
    vision_scoped_config,
    vision_tower_passthrough_inputs,
)
from TraceLens.ModelUtils.ast_analyze import (
    analyze_source,
    is_forward_operation,
    operation_display_label,
)
from TraceLens.ModelUtils.kernel_pipeline import (
    parse_kernel_import,
    _find_symbol_definition,
)
from TraceLens.ModelUtils.model_graph import (
    GraphEdge,
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

# Reserved key separator for per-output-port shape specs. A tuple-unpacked
# split/chunk/unbind publishes one spec per ordinal (its slice shape) under
# ``f"{node_id}{PORT_SPEC_SEP}{ordinal}"`` so the exporter can label each output
# port of the split with the right sub-shape. The null bytes never occur in node
# ids, so these entries never collide with a real node lookup.
_PASS_THROUGH_PORTS = frozenset(
    {"@kernel_port_in", "@kernel_port_out", "@input_mirror", "@output_mirror"}
)
PORT_SPEC_SEP = "\x00port\x00"


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
    # Vision-tower patch/sequence axis. Distinct from the text ``B*S`` so a VLM's
    # image tokens (and the spatial-merge ``Pv/4`` reduction) are not conflated with
    # the language sequence they are scattered into at ``masked_scatter``.
    VISION_PATCH = "Pv"
    # Images (or videos) in the batch -- the rows of a vision grid descriptor
    # ``[Img, 3]``. Distinct from ``Pv``: one image contributes many patches, so
    # counting the grid's rows in patches is what made GLM's ``cu_seqlens`` claim
    # one segment boundary per patch.
    VISION_GRID = "Img"


# Dimensions that vary with the INPUT rather than the checkpoint. Every other
# symbol above is fixed the moment a checkpoint is chosen, so it must resolve to
# a number -- this is what says which letters are legitimately still letters in a
# rendered shape. It is not a guess about any model's spelling: a batch is a
# batch everywhere, which is why this partition can be stated once while the
# NAMES a model reads its constants under have to come from the model.
_RUNTIME_SYMBOLS: frozenset[Symbol] = frozenset(
    {Symbol.BATCH, Symbol.SEQ, Symbol.VISION_PATCH, Symbol.VISION_GRID}
)


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
    dtype: str | None = None
    """Set only when ``__init__`` states it (``torch.zeros(..., dtype=torch.long)``).

    A routing table is int64 and reads as one; left unset, a buffer takes the
    module's working precision like any activation.
    """


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
    conv_geometry: dict[
        str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = field(default_factory=dict)
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
        # (`config.num_local_experts` against `n_routed_experts`, say). The model
        # says which, so read its declaration rather than registering a list of
        # spellings models MIGHT use: such a list has to claim a name globally,
        # and `n_heads` is GLM's sparse indexer head count (`self.n_heads =
        # config.index_n_heads`, 32), not another word for its 64 attention
        # heads. Registering the guess first shadowed the real value.
        for read_name, actual_key in _model_declared_aliases(spec).items():
            value = dims.get(actual_key)
            if isinstance(value, int):
                dims.setdefault(read_name, value)

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


def _called_class_name(func: ast.AST) -> str | None:
    """Class name a constructor call names (``SubClass(...)`` / ``mod.SubClass(...)``)."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _ctor_argument_bindings(
    init_func: ast.FunctionDef,
    call: ast.Call,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> dict[str, DimExpr]:
    """Constructor parameters this call site pins to a resolvable value.

    Maps the call's positional and keyword arguments onto ``init_func``'s parameter
    names and resolves each expression against the OWNER's config and locals, since
    that is the scope the expression is written in. Only arguments that resolve are
    returned; a parameter left to its default is absent, so the submodule's own
    ``__init__`` still decides it.
    """
    params = [arg.arg for arg in init_func.args.args if arg.arg != "self"]
    bound: dict[str, ast.AST] = {}
    for index, value in enumerate(call.args):
        if index < len(params):
            bound[params[index]] = value
    for keyword in call.keywords:
        if keyword.arg in params:
            bound[keyword.arg] = keyword.value
    resolved: dict[str, DimExpr] = {}
    for name, value in bound.items():
        dim = _resolve_dim_expr(
            value, config=config, local_vars=local_vars, context=context
        )
        if dim is not None:
            resolved[name] = dim
    return resolved


@dataclass
class ModuleDimRegistry:
    """Linear and embedding constructor dimensions parsed from modeling AST."""

    linear: dict[tuple[str, str], ModuleLinearSpec] = field(default_factory=dict)
    linear_by_attr: dict[str, ModuleLinearSpec] = field(default_factory=dict)
    #: ``(owner_class, owner_attr, linear_attr) -> spec`` for a submodule whose
    #: constructor ARGUMENTS override what its own ``__init__`` would read from
    #: config. The same class built two ways has two different right answers, so
    #: these cannot live in the per-class table above.
    linear_by_owner: dict[tuple[str, str], ModuleLinearSpec] = field(
        default_factory=dict
    )
    #: ``(owner_attr, linear_attr)`` pairs two owners disagree on, which therefore
    #: identify nothing and must not answer.
    linear_owner_ambiguous: set[tuple[str, str]] = field(default_factory=set)
    #: Resolved ``__init__`` locals per class, kept so a construction site can
    #: resolve its argument expressions against the OWNER's symbols.
    class_locals: dict[str, dict[str, DimExpr]] = field(default_factory=dict)
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

    def known_classes(self) -> set[str]:
        """Class names that own at least one parsed module dim.

        Used to tell a real owning module class (which should scope
        ``(class, attr)`` lookups) from a primitive op label carried in node
        metadata (``View``/``Conv2d``), which must not suppress the attr-only
        fallback for an otherwise-unresolved node.
        """
        names: set[str] = set(self.scalar_by_class)
        for table in (self.linear, self.embedding, self.parameter, self.conv):
            names.update(class_name for class_name, _attr in table)
        return names

    @classmethod
    def from_registry(
        cls,
        class_registry: dict[str, ClassStructure],
        *,
        config: dict[str, Any],
        context: ShapeContext,
        vision_scoped: set[str] | None = None,
        vision_config: dict[str, Any] | None = None,
    ) -> ModuleDimRegistry:
        registry = cls()
        for class_name, structure in class_registry.items():
            init_func = _find_init_function(structure.node)
            if init_func is None:
                continue
            local_vars: dict[str, DimExpr] = {}
            # A vision-scoped class resolves ``config.<attr>`` against the vision
            # sub-config so e.g. ``self.embed_dim = config.hidden_size`` lands on the
            # vision hidden (1024), and ``nn.Conv3d(in_channels, embed_dim, ...)``
            # gets the vision channel counts, not the text stack's.
            class_config = (
                vision_config
                if vision_scoped and vision_config and class_name in vision_scoped
                else config
            )
            registry._walk_init_body(
                init_func.body,
                class_name=class_name,
                config=class_config,
                local_vars=local_vars,
                context=context,
            )
            registry.class_locals[class_name] = dict(local_vars)
            registry._capture_buffer_shapes(
                class_name,
                structure.node,
                config=class_config,
                context=context,
            )

        # Second pass: a submodule built with constructor-argument overrides has
        # different dimensions per INSTANTIATION. ``Glm5NextTextMLP`` is built once
        # bare (its own ``config.intermediate_size``) and once as ``shared_experts``
        # with ``intermediate_size=moe_intermediate_size * n_shared_experts``; a
        # table keyed only by class must report one of those for both. Re-read the
        # submodule's ``__init__`` with the call site's arguments bound and file the
        # result under the owner, leaving the per-class table untouched.
        for parent, structure in class_registry.items():
            init_func = _find_init_function(structure.node)
            if init_func is None:
                continue
            parent_config = (
                vision_config
                if vision_scoped and vision_config and parent in vision_scoped
                else config
            )
            parent_locals = registry.class_locals.get(parent, {})
            for stmt in ast.walk(init_func):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                call = stmt.value
                if not (
                    isinstance(target, ast.Attribute)
                    and _is_self_attr(target)
                    and isinstance(call, ast.Call)
                ):
                    continue
                sub_name = _called_class_name(call.func)
                sub_structure = class_registry.get(sub_name or "")
                if sub_structure is None:
                    continue
                sub_init = _find_init_function(sub_structure.node)
                if sub_init is None:
                    continue
                bindings = _ctor_argument_bindings(
                    sub_init,
                    call,
                    config=parent_config,
                    local_vars=parent_locals,
                    context=context,
                )
                if not bindings:
                    continue
                scoped = cls()
                # Walk into a THROWAWAY context: this extra pass exists only to
                # read dimensions, and must not publish conv geometry or other
                # side effects into the context the real passes share.
                scoped._walk_init_body(
                    sub_init.body,
                    class_name=sub_name,
                    config=parent_config,
                    local_vars=dict(bindings),
                    context=copy.deepcopy(context),
                )
                for (owner_cls, attr), spec in scoped.linear.items():
                    if owner_cls != sub_name:
                        continue
                    key = (target.attr, attr)
                    previous = registry.linear_by_owner.get(key)
                    if previous is not None and previous != spec:
                        registry.linear_owner_ambiguous.add(key)
                    registry.linear_by_owner[key] = spec
        return registry

    def _capture_buffer_shapes(
        self,
        class_name: str,
        class_node: ast.ClassDef,
        *,
        config: dict[str, Any],
        context: ShapeContext,
    ) -> None:
        """Register 1-D buffers assigned from ``torch.arange`` (RoPE ``inv_freq``).

        Handles ``self.<attr> = nn.Buffer(x)`` and ``self.register_buffer("attr", x)``
        when ``x`` traces to a ``torch.arange`` (through local aliases and one
        same-class method hop). General; only fires when a concrete length resolves,
        so non-arange buffers are left untouched.
        """
        init_func = _find_init_function(class_node)
        if init_func is None:
            return
        ast_locals = _function_ast_locals(init_func)
        dim_locals = _resolved_scalar_locals(init_func, config=config, context=context)
        for stmt in ast.walk(init_func):
            attr: str | None = None
            source: ast.AST | None = None
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if _call_class_name(stmt.value) == "Buffer" and stmt.value.args:
                    self_targets = [
                        target
                        for target in stmt.targets
                        if isinstance(target, ast.Attribute) and _is_self_attr(target)
                    ]
                    if self_targets:
                        attr = self_targets[0].attr
                        source = stmt.value.args[0]
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if (
                    _call_class_name(call) == "register_buffer"
                    and len(call.args) >= 2
                    and isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, str)
                ):
                    attr = call.args[0].value
                    source = call.args[1]
            if attr is None or source is None:
                continue
            if (class_name, attr) in self.parameter:
                continue
            length = _resolve_buffer_length(
                source,
                class_node=class_node,
                config=config,
                context=context,
                ast_locals=ast_locals,
                dim_locals=dim_locals,
            )
            if isinstance(length, int) and length > 0:
                spec = ModuleParameterSpec(shape=(length,))
                self.parameter[(class_name, attr)] = spec
                self.parameter_by_attr.setdefault(attr, spec)
                continue
            # Not a 1-D ``arange``. A buffer built by a plain constructor states
            # its own shape and dtype -- ``nn.Buffer(torch.zeros(vocab_size,
            # top_k, dtype=torch.long))`` is the token -> expert routing table,
            # and without this it renders as a scalar in the module's working
            # precision, so the lookup that reads it loses both its rank and its
            # integer-ness and every consumer downstream inherits that.
            built = _constructor_buffer_spec(
                source,
                config=config,
                context=context,
                dim_locals=dim_locals,
                self_dims=_resolved_self_dims(
                    init_func, config=config, context=context, dim_locals=dim_locals
                ),
            )
            if built is not None:
                self.parameter[(class_name, attr)] = built
                self.parameter_by_attr.setdefault(attr, built)

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
                        class_name,
                        target,
                        stmt.value,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                self._record_assignment(
                    class_name,
                    stmt.target,
                    stmt.value,
                    config=config,
                    local_vars=local_vars,
                    context=context,
                )
            elif isinstance(stmt, ast.If):
                # Evaluate the condition against config and local_vars.
                branch = _eval_config_condition(
                    stmt.test, config=config, local_vars=local_vars
                )
                if branch is True:
                    self._walk_init_body(
                        stmt.body,
                        class_name=class_name,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
                elif branch is False:
                    self._walk_init_body(
                        stmt.orelse,
                        class_name=class_name,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
                else:
                    # Cannot evaluate condition — process both branches so
                    # we don't miss assignments.  Later branch wins.
                    self._walk_init_body(
                        stmt.body,
                        class_name=class_name,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
                    self._walk_init_body(
                        stmt.orelse,
                        class_name=class_name,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
            elif isinstance(stmt, (ast.For, ast.While, ast.With)):
                self._walk_init_body(
                    stmt.body,
                    class_name=class_name,
                    config=config,
                    local_vars=local_vars,
                    context=context,
                )
            elif isinstance(stmt, ast.Try):
                self._walk_init_body(
                    stmt.body,
                    class_name=class_name,
                    config=config,
                    local_vars=local_vars,
                    context=context,
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
            elif isinstance(value, (ast.List, ast.Tuple)):
                # List locals feed conv geometry, e.g.
                # ``kernel_size = [temporal_patch_size, patch_size, patch_size]``.
                int_tuple = _resolve_int_tuple(
                    value, config=config, local_vars=local_vars, context=context
                )
                if int_tuple is not None:
                    local_vars[target.id] = int_tuple
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
        # Boolean reductions: they reduce an axis like any other reduction, and
        # the answer is bool whatever went in.
        "any",
        "all",
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
        # ``scatter_add`` writes into a copy of its first operand, so the output
        # keeps that operand's shape and dtype like any other element-wise write.
        "scatter add",
    }
)
# Display labels (lowercased) for pointwise ops whose *first* operand is the
# tensor being written into (``Tensor.masked_fill(mask, value)`` and friends).
# The result is shaped and typed exactly like operand 0; the trailing
# mask/index/source operands only broadcast or address into it and must never
# win the ``_broadcast_rank`` vote (a wide bool mask would otherwise turn a
# float write into a bool tensor of the mask's shape).
_FIRST_OPERAND_WRITE_LABELS = frozenset(
    {
        "masked fill",
        "masked scatter",
        "scatter",
        "scatter add",
        "index add",
        "copy",
    }
)
# Display labels (lowercased) for element-wise comparisons. Like a pointwise op
# they keep the widest operand's shape, but the result dtype is always boolean.
_COMPARISON_LABELS = frozenset(
    {
        "greater",
        "greater equal",
        "less",
        "less equal",
        "equal",
        "not equal",
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


def _resolve_op_arg(raw: str, dims: dict[str, Any]) -> int | list[int] | None:
    """Resolve a recorded call-argument string to a concrete int or list of ints.

    Handles a single value (``4``, ``-1``, ``head_dim``) via
    :func:`_resolve_op_int` (config-name lookup included) and a comma-separated
    tuple/list (``(2, 4)``, ``(head_dim, 4)``) as a list of ints with at most one
    unresolved ``-1`` placeholder. Anything else (a dtype, an einsum equation, a
    raw-op name) does not resolve and returns ``None`` -- so a caller can simply
    resolve *every* detail and keep only the ones that turn into real arguments,
    with no hand-maintained detail-key allowlist.
    """
    if raw is None:
        return None
    value = _resolve_op_int(raw, dims)
    if value is not None:
        return value
    inner = raw.strip().strip("()[]")
    if "," not in inner:
        return None
    parts = [part.strip() for part in inner.split(",") if part.strip()]
    if not parts:
        return None
    resolved: list[int] = []
    for part in parts:
        item = _resolve_op_int(part, dims)
        resolved.append(item if item is not None else -1)
    if resolved.count(-1) > 1:
        return None
    return resolved


def _op_scalar_detail_args(
    details: Sequence[str], dims: dict[str, Any]
) -> list[tuple[str, int | list[int]]]:
    """Ordered ``(name, value)`` scalar arguments recovered from a node's call
    details. Each detail line ``key: value`` is resolved with
    :func:`_resolve_op_arg`; only the lines that resolve to an int / list of ints
    survive, so descriptive details (``raw_op:``, ``dtype:``, ``mutates:`` …) drop
    out naturally without a curated key set."""
    args: list[tuple[str, int | list[int]]] = []
    for item in details:
        text = str(item).strip()
        if ":" not in text:
            continue
        key, _, raw = text.partition(":")
        value = _resolve_op_arg(raw.strip(), dims)
        if value is not None:
            args.append((key.strip(), value))
    return args


def _resolve_meta_op_callable(torch: Any, name: str) -> Any:
    """Resolve an op name to a real torch callable, or ``None``.

    Pure attribute lookup across the ``torch`` / ``torch.Tensor`` /
    ``torch.nn.functional`` namespaces and the aten operator registry -- no
    per-op allowlist. The name is the one the model itself calls (the ``raw_op``
    recovered from its forward source, or the op's display label as a fallback),
    so ``div``/``cumsum``/``split``/``unbind``/... all resolve to the callable
    whose meta execution gives the ground-truth output shape.
    """
    name = str(name or "").strip()
    if not name:
        return None
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    for owner in (torch, torch.Tensor, functional):
        if owner is None:
            continue
        candidate = getattr(owner, name, None)
        if callable(candidate):
            return candidate
    try:
        packet = getattr(torch.ops.aten, name, None)
    except Exception:  # noqa: BLE001
        packet = None
    if packet is not None and callable(packet):
        return packet
    return None


def _op_positional_params(
    torch: Any, fn: Any, name: str
) -> tuple[list[tuple[str, bool]], bool] | None:
    """Ordered positional parameters ``(name, has_default)`` of an op, plus a
    ``has_var_positional`` flag.

    Read from ``inspect.signature`` when available (pure-Python ops such as
    ``torch.split``); for C-builtins with no introspectable signature
    (``chunk``/``unbind``/``unflatten``/...) fall back to the op's aten operator
    schema, which lists ordered argument names and defaults. Returns ``None`` when
    neither source can describe the op -- the caller then only tries a bare call.
    This is not an op allowlist: it reads whatever real parameters the resolved
    callable declares.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        params = [
            (param.name, param.default is not inspect.Parameter.empty)
            for param in sig.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_var = any(
            param.kind == inspect.Parameter.VAR_POSITIONAL
            for param in sig.parameters.values()
        )
        if params:
            return params, has_var
    try:
        schemas = torch._C._jit_get_schemas_for_operator(f"aten::{name}")
    except Exception:  # noqa: BLE001
        schemas = None
    if schemas:
        schema = schemas[0]
        params = []
        for arg in getattr(schema, "arguments", []):
            if getattr(arg, "kwarg_only", False):
                continue
            try:
                has_default = arg.has_default_value()
            except Exception:  # noqa: BLE001
                has_default = getattr(arg, "default_value", None) is not None
            params.append((str(arg.name), bool(has_default)))
        if params:
            return params, False
    return None


def _bind_meta_op_args(
    params: list[tuple[str, bool]],
    has_var_positional: bool,
    metas: list[Any],
    scalar_args: list[tuple[str, int | list[int]]],
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Bind meta tensors + recovered scalar args to an op's positional parameters.

    Tensor operands fill the leading positional parameters in order (torch ops put
    their tensor operands first); every remaining positional parameter is a scalar
    slot filled in two passes -- by name first (exact or prefix match, so
    ``split_size`` binds ``split_size_or_sections``), then positionally from the
    still-unused scalar values in source order (so a ``chunks`` parameter still
    receives the recorded count even though the detail key was ``split_size``).
    Name matching runs globally before positional fallback, so a value that names
    a later parameter is never stolen by an earlier unnamed one. Returns ``None``
    when a required parameter cannot be bound -- the caller then falls back to a
    bare positional call.
    """

    def _name_matches(param: str, key: str) -> bool:
        param = param.strip().lower()
        key = key.strip().lower()
        return bool(key) and (
            param == key or param.startswith(key) or key.startswith(param)
        )

    num_tensor = min(len(metas), len(params))
    bound: list[Any] = list(metas[:num_tensor])
    scalar_slots = params[num_tensor:]
    used = [False] * len(scalar_args)
    values: list[Any] = [None] * len(scalar_slots)
    # Pass 1: name match.
    for slot_index, (pname, _default) in enumerate(scalar_slots):
        for index, (key, val) in enumerate(scalar_args):
            if not used[index] and _name_matches(pname, key):
                values[slot_index], used[index] = val, True
                break
    # Pass 2: positional fill for still-unbound scalar slots.
    for slot_index in range(len(scalar_slots)):
        if values[slot_index] is not None:
            continue
        for index, (_key, val) in enumerate(scalar_args):
            if not used[index]:
                values[slot_index], used[index] = val, True
                break
    for slot_index, (_pname, has_default) in enumerate(scalar_slots):
        if values[slot_index] is None:
            if has_default:
                # Leave this and every later positional slot to their defaults.
                break
            return None
        bound.append(values[slot_index])
    leftover = metas[num_tensor:]
    if leftover:
        if not has_var_positional:
            return None
        bound.extend(leftover)
    return tuple(bound), {}


def _neutral_scalar_args(torch: Any, name: str) -> list[tuple[str, Any]]:
    """Non-tensor parameters of ``aten::<name>``, filled with neutral values.

    Read from the op's real schema, so nothing here names an operation. Used
    only to ask whether an op preserves its input's shape, where the value of a
    dropout probability or a training flag cannot change the answer.
    """
    try:
        schemas = torch._C._jit_get_schemas_for_operator(f"aten::{name}")
    except Exception:  # noqa: BLE001
        return []
    neutral = {"float": 0.0, "bool": False, "int": 1}
    for schema in schemas:
        args: list[tuple[str, Any]] = []
        for argument in schema.arguments:
            kind = str(argument.type)
            if kind == "Tensor":
                continue
            if kind not in neutral:
                args = []
                break
            args.append((argument.name, neutral[kind]))
        if args:
            return args
    return []


def _run_meta_op(
    torch: Any,
    fn: Any,
    name: str,
    metas: list[Any],
    scalar_args: list[tuple[str, int | list[int]]],
) -> Any:
    """Execute ``fn`` on the meta device and return its output, or ``None``.

    Tries a parameter-bound call first (meta tensors + recovered scalar args,
    from the op's real signature or aten schema), then a bare positional call for
    element-wise ops that take only their operands. Any failure (data-dependent
    op, custom kernel, unbindable arg) returns ``None`` so the caller passes
    through -- meta execution never fabricates a shape it could not really
    produce.
    """
    strategies: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    params = _op_positional_params(torch, fn, name)
    if params is not None:
        bound = _bind_meta_op_args(params[0], params[1], metas, scalar_args)
        if bound is not None:
            strategies.append(bound)
    strategies.append((tuple(metas), {}))
    for args, kwargs in strategies:
        try:
            with torch.device("meta"):
                out = fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            continue
        if out is not None and _first_tensor_shape(out) is not None:
            return out
    return None


def _normalize_op_name(name: Any) -> str:
    """Normalize an op name for cross-source matching (must stay identical to
    ``meta_trace._norm_op`` so AST ids and FX node names join)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------- #
# Dynamic operand-arity resolution.
#
# An operation's tensor-operand contract is read from the op's *real function
# parameters*, never from a hardcoded op-name list: the raw op name is the one
# recovered from the model's own forward source (stamped as the ``raw_op`` node
# attr/detail by the extractor), and its arity comes from introspecting that
# callable. ``inspect.signature`` is tried first -- it reads any annotated
# pure-Python op -- and the aten operator schema is the fallback for the
# C-builtin torch ops (``transpose``/``cat``/``view``/``squeeze``/...) that have
# no introspectable Python signature. An op whose parameters cannot be resolved
# is simply skipped (no false positives). Shared by the type-check pass
# (``type_check.py``) and the wiring pass (``computation_graph.py``), which both
# need "how many real tensor operands does this op take" without guessing from
# the op's name.
# --------------------------------------------------------------------------- #

# Parameter type strings that denote a single tensor operand vs an unbounded
# tensor *list* (the variadic ``cat``/``stack`` contract). Matched against both
# ``inspect`` annotations and aten schema argument types.
_TENSOR_ARG_TYPES = frozenset({"Tensor", "Optional[Tensor]", "Tensor?"})
_TENSOR_LIST_ARG_TYPES = frozenset({"List[Tensor]", "Tensor[]"})


def _annotation_tensor_kind(annotation: Any) -> str:
    """Classify an ``inspect`` parameter annotation: ``"tensor"`` / ``"list"`` / ""."""
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", None) or str(annotation)
    )
    text = text.replace("torch.", "").replace(" ", "")
    if text in _TENSOR_ARG_TYPES or text == "Tensor":
        return "tensor"
    if text in _TENSOR_LIST_ARG_TYPES or (
        text.startswith(("List[", "Sequence[", "Tuple[", "Iterable["))
        and "Tensor" in text
    ):
        return "list"
    if text.startswith("Optional[") and "Tensor" in text and "List" not in text:
        return "tensor"
    return ""


def _ceiling_via_inspect(name: str) -> tuple[int | None, bool] | None:
    """``(max_tensor_operands, is_variadic)`` from ``inspect``, or ``None`` if it
    cannot type the op (unresolvable callable, no signature, or no annotations)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in the pipeline
        return None
    fn = None
    for owner in (torch.Tensor, torch):
        candidate = getattr(owner, name, None)
        if candidate is not None:
            fn = candidate
            break
    if fn is None:
        return None
    try:
        signature = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    count = 0
    variadic = False
    saw_annotation = False
    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            variadic = True
            continue
        if param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is inspect.Parameter.empty:
            continue
        saw_annotation = True
        kind = _annotation_tensor_kind(param.annotation)
        if kind == "list":
            variadic = True
        elif kind == "tensor":
            count += 1
    if not saw_annotation:
        # ``inspect`` gave a signature but no types (e.g. ``torch.split``): it
        # cannot distinguish tensor operands from scalar args -- defer to the schema.
        return None
    return (None, True) if variadic else (count, False)


def _ceiling_via_aten_schema(name: str) -> tuple[int | None, bool]:
    """``(max_tensor_operands, is_variadic)`` from the aten operator schema.

    Counts non-``out`` (positional / non-kwarg-only) ``Tensor``/``Optional[Tensor]``
    arguments; a ``List[Tensor]`` argument marks the op variadic (unbounded, e.g.
    ``cat``). Unknown ops (custom free functions with no aten schema) return
    ``(None, False)`` -> skipped by the caller.
    """
    try:
        import torch

        schemas = torch._C._jit_get_schemas_for_operator(f"aten::{name}")
    except Exception:
        return None, False
    if not schemas:
        return None, False
    counts: list[int] = []
    any_variadic = False
    for schema in schemas:
        count = 0
        variadic = False
        for arg in getattr(schema, "arguments", []):
            if getattr(arg, "kwarg_only", False):
                continue
            arg_type = str(getattr(arg, "type", ""))
            if arg_type in _TENSOR_LIST_ARG_TYPES:
                variadic = True
            elif arg_type in _TENSOR_ARG_TYPES:
                count += 1
        if variadic:
            any_variadic = True
        else:
            counts.append(count)
    if any_variadic:
        return None, True
    return (max(counts) if counts else None), False


@functools.lru_cache(maxsize=None)
def _operand_ceiling(raw_op: str) -> tuple[int | None, bool]:
    """``(max_tensor_operands, is_variadic)`` for an op, resolved from its real
    parameters. ``(None, False)`` means the arity could not be determined (skip);
    ``(None, True)`` means an unbounded tensor-list op (``cat``/``stack``)."""
    name = str(raw_op or "").strip()
    if not name:
        return None, False
    via_inspect = _ceiling_via_inspect(name)
    if via_inspect is not None:
        return via_inspect
    return _ceiling_via_aten_schema(name)


def _required_names_via_aten_schema(name: str) -> tuple[str, ...] | None:
    """Ordered names of the REQUIRED positional tensor args of an aten op's schema.

    A ``Tensor``/``Optional[Tensor]`` positional argument that carries **no**
    default is required (sdpa's ``query``/``key``/``value``); one with a default
    (``attn_mask=None``) is optional and excluded. A ``List[Tensor]`` argument
    (``cat``) is variadic -- no fixed required-name set -- so that schema is
    skipped. ``None`` when the op has no aten schema."""
    try:
        import torch

        schemas = torch._C._jit_get_schemas_for_operator(f"aten::{name}")
    except Exception:
        return None
    if not schemas:
        return None
    best: tuple[str, ...] | None = None
    for schema in schemas:
        names: list[str] = []
        variadic = False
        for arg in getattr(schema, "arguments", []):
            if getattr(arg, "kwarg_only", False):
                continue
            arg_type = str(getattr(arg, "type", ""))
            if arg_type in _TENSOR_LIST_ARG_TYPES:
                variadic = True
                break
            if arg_type in _TENSOR_ARG_TYPES:
                has_default = (
                    arg.has_default_value()
                    if hasattr(arg, "has_default_value")
                    else False
                )
                if not has_default:
                    arg_name = str(getattr(arg, "name", "")).strip()
                    if arg_name:
                        names.append(arg_name)
        if variadic:
            continue
        cand = tuple(names)
        if best is None or len(cand) > len(best):
            best = cand
    return best


def _required_names_via_inspect(name: str) -> tuple[str, ...] | None:
    """Ordered names of the REQUIRED (no-default, annotated tensor) positional
    parameters of an annotated pure-Python torch op, or ``None`` if it cannot be
    typed (unresolvable callable, no signature, or unannotated)."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch always present in the pipeline
        return None
    fn = None
    for owner in (torch.Tensor, torch):
        candidate = getattr(owner, name, None)
        if candidate is not None:
            fn = candidate
            break
    if fn is None:
        return None
    try:
        signature = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    names: list[str] = []
    saw_annotation = False
    for param in signature.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.annotation is inspect.Parameter.empty:
            continue
        saw_annotation = True
        if (
            _annotation_tensor_kind(param.annotation) == "tensor"
            and param.default is inspect.Parameter.empty
        ):
            names.append(param.name)
    if not saw_annotation:
        return None
    return tuple(names)


@functools.lru_cache(maxsize=None)
def _required_tensor_operand_names(raw_op: str) -> tuple[str, ...]:
    """Ordered names of an op's REQUIRED (no-default) positional tensor operands.

    Resolved from the op's real function parameters -- ``inspect`` for an
    annotated pure-Python op, else the aten operator schema for a C-builtin torch
    op -- never a hardcoded op-name -> operand map. Used by the type-check's
    kernel operand-arity coverage check to learn how many DISTINCT tensor operands
    a kernel structurally requires (sdpa: ``query``/``key``/``value``; the optional
    ``attn_mask`` is excluded). Empty tuple when the op has no resolvable signature
    (a custom free function), so the caller skips the check (no false positive)."""
    name = str(raw_op or "").strip()
    if not name:
        return ()
    via_inspect = _required_names_via_inspect(name)
    if via_inspect is not None:
        return via_inspect
    return _required_names_via_aten_schema(name) or ()


# Trailing ``@op_l{line}_c{col}_{name}[:idx]`` token of a graph node id, with
# everything before it captured as the block-instance prefix.
_LAST_OP_ID_RE = re.compile(
    r"^(?P<prefix>.*):@op_l(?P<line>\d+)_c(?P<col>\d+)_(?P<name>[a-z0-9_]+?)"
    r"(?::(?P<idx>\d+))?$"
)


def _owning_attr_candidates(node: Any, root: Any) -> list[str]:
    """Submodule attrs that could own a constant leaf, nearest first.

    A constant reached through an inline submodule frame names that submodule in
    its own id (``seq:2:q_norm:@op_...:const:weight``). One reached directly in a
    section body does not -- there the owning attr is the section being inferred,
    which only ``root`` knows (``input_layernorm``). Both are offered, id segments
    first, and the caller keeps a candidate only when it resolves unambiguously,
    so the non-module segments this inevitably includes simply match nothing.
    """
    candidates: list[str] = []
    for segment in reversed(re.split(r"[:/]", str(getattr(node, "id", "")))):
        # A purely numeric segment is a POSITION in the id (``seq:7``, the ``:0``
        # ordinal of an op), not the attribute of an owning module. It must not
        # be offered: ``0`` is also a real ModuleList entry name, so ``0.weight``
        # suffix-matches some list member's parameter and -- being a single,
        # unambiguous hit -- wins over the actual owner further along. That is
        # how every Kimi ``input_layernorm`` weight came to report a square
        # [4096, 4096] instead of its own [7168].
        if segment.isdigit():
            continue
        if segment and not segment.startswith("@") and segment not in candidates:
            candidates.append(segment)
    root_attr = getattr(root, "attr_name", None)
    if root_attr and str(root_attr) not in candidates:
        candidates.append(str(root_attr))
    return candidates


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
        # Vision-tower disambiguation: every class instantiated under the tower is
        # constructed with the nested ``vision_config`` (different ``hidden_size``,
        # plus ``in_channels``/``patch_size`` that live only there). Resolve those
        # classes' ``config.<attr>`` against the vision overlay; the text path is
        # untouched (empty set for text-only models).
        self._vision_scoped: set[str] = vision_scoped_classes(spec)
        self._vision_config: dict[str, Any] = (
            vision_scoped_config(spec) if self._vision_scoped else {}
        )
        found_tower = find_vision_tower(spec)
        self._vision_tower_class: str | None = found_tower[1] if found_tower else None
        self._vision_patch_flat: int | None = _vision_patch_flat_dim(
            self._vision_config
        )
        # Frames whose input is a grid DESCRIPTOR the model was handed, not the
        # activation flowing down the tower. Their ops have no predecessor inside
        # the graph, so without this they fall back to the section activation and
        # compute the patch geometry: GLM's ``cu_seqlens`` then reports one
        # segment boundary per patch. Keyed by the frame's attr name, which every
        # op id underneath carries. See ``_descriptor_frame_specs``.
        self._descriptor_frames: dict[str, TensorSpec] = _descriptor_frame_specs(spec)
        # What the caller hands the section currently being inferred, when the
        # section is a method expansion with no caller context of its own. Last
        # resort for its ``@input``, below every authoritative seed.
        self._caller_entry_spec: TensorSpec | None = None
        self.module_dims = module_dims or ModuleDimRegistry.from_registry(
            spec.class_registry,
            config=spec.raw_config or {},
            context=self.context,
            vision_scoped=self._vision_scoped,
            vision_config=self._vision_config,
        )
        # The patch-embed class is the vision-scoped class that consumes the raw
        # pixel patches: it owns a conv whose ``in_channels`` matches the config's
        # image ``in_channels`` (3 for RGB). Seeding *its* forward input with the
        # flat patch shape ``[Pv, C*T*P*P]`` lets the patch-embed views + Conv3d
        # resolve concretely instead of collapsing to a language ``(B, S, H)``.
        self._vision_patch_embed_class: str | None = self._detect_patch_embed_class()
        self._vision_hidden: int | None = (
            _int_dim(self._vision_config.get("hidden_size"))
            if self._vision_config
            else None
        )
        # Active activation geometry for the section currently being inferred.
        # The language default is ``(B, S)`` + text ``hidden``; while inferring a
        # vision-scoped section (root class in ``_vision_scoped``) it switches to a
        # single patch axis ``(Pv,)`` + vision ``hidden``. Every ``(B, S, H)``
        # fallback stamp reads these, so a disconnected vision op (no inputs) lands
        # on ``(Pv, 1024)`` instead of the text sequence ``(B, S, 4096)``.
        self._active_seq_axes: tuple[str, ...] = (
            Symbol.BATCH.value,
            Symbol.SEQ.value,
        )
        self._active_hidden: Any = self.context.dims.get(
            Symbol.HIDDEN.value, Symbol.HIDDEN.value
        )
        # Per-graph @input overrides (port label -> spec), populated by callers that
        # know a section's true upstream boundary shape. Empty for the default path.
        self._entry_specs: dict[str, TensorSpec] = {}
        # A vision rotary embedding consumes ``position_ids`` as
        # ``(total_tokens, N)`` where N is the number of coordinate axes (h,w -> 2;
        # t,h,w -> 3). Without an authoritative seed the boundary inherits the flat
        # patch geometry ([Pv, 1176]) and poisons the whole freq chain. Introspect
        # N from source (the rotary recomposition selects one frequency slice per
        # axis) and seed the ``position_ids`` @input with ``[Pv, N]``. General:
        # only for models with a vision tower, driven off the source, no
        # class-name checks.
        # Authoritative ``[Pv, N]`` seed for the vision rotary ``position_ids``,
        # plus the rotary forward's source-line range. The rotary is inlined into
        # the vision tower forward (no ``@input`` boundary of its own), so besides
        # the @input override below we also seed the internal op that reads the raw
        # ``position_ids`` parameter — the one whose port carries ``position_ids``
        # and whose ``@op_l<line>`` sits inside the rotary forward. See
        # ``_gather_input_specs``.
        self._vision_posid_seed: tuple[TensorSpec, int, int] | None = None
        if self._vision_tower_class is not None and self._vision_scoped:
            seed = self._vision_position_ids_seed()
            if seed is not None:
                axis, start_line, end_line = seed
                self._entry_specs["position_ids"] = TensorSpec(
                    shape=(Symbol.VISION_PATCH.value, axis), dtype="int64"
                )
                self._vision_posid_seed = (
                    TensorSpec(shape=(Symbol.VISION_PATCH.value, axis), dtype="int64"),
                    start_line,
                    end_line,
                )
        # @input boundary node ids this graph resolved via an authoritative
        # ``_entry_spec_for`` seed/override. A root-less subgraph recursion must
        # not clobber these with its default activation spec (see infer_model_graph).
        self._entry_seeded_ids: set[str] = set()
        # ``@output`` boundaries this graph resolved WITH its class context. The
        # root-less subgraph recursion below re-reads the same ids with no class
        # to resolve against, and must not clobber them -- the same protection
        # ``_entry_seeded_ids`` gives the entry side.
        self._rooted_output_ids: set[str] = set()
        # What each module class RETURNS, learned when that class is inferred in
        # its own context. A module drawn as a subgraph in a parent graph is
        # sized before its own body is walked, so without this it falls back to
        # passing its input through. Kept per class and used only when every
        # instance agrees, so a class built at two widths reports neither.
        self._module_output_specs: dict[str, set[tuple[Any, str | None]]] = {}
        self._module_resolved_ids: set[str] = set()
        self._tensor_names: dict[str, str] = {}
        self._tensor_specs: dict[str, TensorSpec] = {}
        self._tiling_slot_ids: set[str] = set()
        self._boundary_shapes: dict[str, tuple] = {}
        # Shape of the tensor each inlined frame was HANDED, keyed by the
        # frame's id prefix. A frame's forward can reshape by its own input
        # parameter, which is a different tensor from the one being
        # reshaped, and parameter names like `x` repeat across frames.
        self._frame_entry_shapes: dict[str, tuple] = {}
        self._last_input_sources: list[str] = []
        self._owner_classes: dict[int, dict[str, str]] = {}
        # CPython reuses an object's address once it is freed, so a cache keyed
        # by ``id()`` can serve one object's entry to an unrelated later one --
        # the owner-class map of a discarded block answering for whatever block
        # next lands at that address, which silently resolves a parameter
        # against the wrong class. Holding a reference keeps each address
        # unique for as long as its key is in use.
        self._owner_class_refs: list[Any] = []
        self._forward_input_spec_refs: list[Any] = []
        # Specs carrying the activation a block's forward receives.
        self._forward_input_specs: set[int] = set()
        # Guard against infinite recursion during forward introspection.
        self._introspecting: set[str] = set()
        # Meta-device traced shapes (module path -> symbolic shape).
        self._meta_shapes: dict[str, TensorSpec] = {}
        # Meta-device forward-parameter *input* shapes, keyed by parameter name,
        # restricted to parameters that are globally shape-consistent (see
        # ``trace_meta_input_specs``). Used to resolve an ``@input`` boundary that
        # would otherwise fall back to the generic ``(B, S, hidden)`` default —
        # e.g. an ``attention_mask`` that is genuinely ``[B, S]``.
        self._meta_input_specs: dict[str, TensorSpec] = {}
        # Class-scoped meta-device forward-parameter *input* shapes, keyed by
        # ``(receiving module class name, parameter name)``. Resolves a boundary
        # whose name is globally ambiguous but locally definite — e.g.
        # ``attention_mask`` is a 4-D causal mask on the decoder attention yet a
        # flat ``[B, S]`` padding mask on the sparse-attention indexer.
        self._meta_input_specs_by_class: dict[tuple[str, str], TensorSpec] = {}
        # Whether the meta forward-parameter input specs above have been populated.
        # ``load_meta_shapes`` fills them eagerly (CLI path); a direct
        # ``build_merged_model_graph`` never calls it, so ``boundary_input_spec``
        # lazily traces them on first use so ``@input`` boundaries are sized from
        # ground truth in both paths.
        self._meta_input_specs_loaded: bool = False
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
        # Lazily-harvested meta parameter/buffer index (shape + dtype), used to
        # size a materialized ``constant`` leaf (a buffer like the rotary
        # ``inv_freq``). ``False`` = not yet built; ``None`` = build failed/unavailable.
        self._meta_tensor_index: Any = False

    def _ensure_meta_tensor_index(self) -> Any:
        if self._meta_tensor_index is not False:
            return self._meta_tensor_index
        self._meta_tensor_index = None
        if self._meta_checkpoint is None:
            return None
        try:
            from TraceLens.ModelUtils.meta_trace import harvest_meta_tensors

            self._meta_tensor_index = harvest_meta_tensors(self._meta_checkpoint)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Meta-tensor harvest failed: %s", exc)
        return self._meta_tensor_index

    def constant_spec(
        self,
        node: ModelGraphNode,
        *,
        root: BlockNode | None,
        names: Sequence[str],
    ) -> TensorSpec | None:
        """Shape + dtype of a materialized ``constant`` leaf (buffer / learned weight).

        Prefers the harvested meta tensor index (has a real dtype and covers
        buffers) keyed by ``(owner_class, leaf)`` then by qualified-name suffix,
        and falls back to :meth:`_lookup_parameter_spec` (which also covers
        ``torch.arange``-derived buffers registered from the AST) with the model
        dtype when meta is unavailable.
        """
        index = self._ensure_meta_tensor_index()
        if index is not None:
            # Qualify the constant with the submodule that OWNS it before trying
            # anything class-wide. ``by_class_attr`` keeps one entry per
            # (class, attr), so a norm class instantiated at several widths -- a
            # head-dim ``q_norm`` at 128 and an ``input_layernorm`` at 6144 --
            # collapses to whichever width was harvested, and then every instance
            # reports that one. The owning attr separates them, and repeated
            # decoder layers all agree, so matching many entries is fine as long
            # as they concur on shape and dtype.
            for attr in _owning_attr_candidates(node, root):
                for name in names:
                    leaf = str(name).split(".")[-1].strip()
                    qualified = f"{attr}.{leaf}"
                    agreed = {
                        (tuple(spec.shape), spec.dtype)
                        for full, spec in index.by_qualified.items()
                        if full == qualified or full.endswith("." + qualified)
                    }
                    if len(agreed) == 1:
                        shape, dtype = next(iter(agreed))
                        return TensorSpec(shape=shape, dtype=dtype)
            owner = self._owner_class_name(node, root)
            for name in names:
                leaf = str(name).split(".")[-1].strip()
                if owner is not None:
                    spec = index.by_class_attr.get((owner, leaf))
                    if spec is not None:
                        return TensorSpec(shape=tuple(spec.shape), dtype=spec.dtype)
            for name in names:
                qualified = str(name).strip()
                matches = [
                    spec
                    for full, spec in index.by_qualified.items()
                    if full == qualified or full.endswith("." + qualified)
                ]
                # A suffix match only identifies a tensor when it lands on exactly
                # one. A bare leaf like ``weight`` suffix-matches EVERY parameter in
                # the model, so taking the first would hand a text RMSNorm whichever
                # ``*.weight`` happened to be enumerated first -- e.g. the vision
                # patch-embed conv's [1280, 3, 2, 14, 14]. Ambiguity falls through to
                # the parameter lookup below, which resolves the owning module's own
                # dimensions instead of guessing.
                if len(matches) == 1:
                    spec = matches[0]
                    return TensorSpec(shape=tuple(spec.shape), dtype=spec.dtype)
                if len(matches) > 1:
                    _log.debug(
                        "%r suffix-matches %d meta tensors; resolving %s from its "
                        "owning module instead",
                        qualified,
                        len(matches),
                        node.id,
                    )
        parameter = self._lookup_parameter_spec(node, root=root, names=names)
        if parameter is not None:
            has_weight = any("weight" in str(n).lower() for n in names)
            # A buffer whose ``__init__`` states its dtype keeps it: a routing
            # table is int64, and reading it as an activation loses the
            # integer-ness every index consumer depends on.
            param_dtype = parameter.dtype or (
                self.context.weight_dtype(str(node.id))
                if has_weight
                else self.context.dtype
            )
            return TensorSpec(shape=parameter.shape, dtype=param_dtype)
        return None

    def _detect_patch_embed_class(self) -> str | None:
        """Vision-scoped class whose conv consumes the raw image channels."""
        if not self._vision_scoped:
            return None
        in_channels = _int_dim(self._vision_config.get("in_channels"))
        if in_channels is None:
            return None
        for (class_name, _attr), conv in self.module_dims.conv.items():
            if class_name not in self._vision_scoped:
                continue
            if _int_dim(conv.in_channels) == in_channels:
                return class_name
        return None

    def _active_hidden_shape(self) -> tuple[Any, ...]:
        """The current section's default activation *shape tuple*.

        ``(B, S, H)`` for the text stack; ``(Pv, Hv)`` while a vision-scoped
        section is being inferred. Mirrors the module-level ``_default_hidden_shape``
        but honours the active vision geometry.
        """
        return (*self._active_seq_axes, self._active_hidden)

    def _active_flattened_seq(self) -> str:
        """Product symbol for the flattened sequence axis of the active section.

        ``B*S`` for the text stack, ``Pv`` for the single vision patch axis. Used
        when a ``-1`` reshape collapses the leading axes and no per-dim symbol
        survives, so the vision tower never inherits the text ``B*S`` symbol.
        """
        return "*".join(str(axis) for axis in self._active_seq_axes)

    def _activation_spec(
        self, dtype: str | None = None, *, hidden: Any = None
    ) -> TensorSpec:
        """The current section's default activation shape.

        ``(B, S, H)`` for the text stack; ``(Pv, Hv)`` while a vision-scoped
        section is being inferred (see ``_active_seq_axes``/``_active_hidden``).
        """
        h = hidden if hidden is not None else self._active_hidden
        return TensorSpec(
            shape=(*self._active_seq_axes, h), dtype=dtype or self.context.dtype
        )

    def _vision_position_ids_seed(self) -> tuple[int, int, int] | None:
        """Resolve the vision rotary ``position_ids`` shape ``[Pv, N]`` from source.

        Returns ``(N, forward_start_line, forward_end_line)`` for the vision rotary
        embedding, or ``None`` when no such class exists.

        ``N`` (coordinate-axis size) is read structurally: the vision rotary
        embedding recomposes its frequencies by selecting one slice per coordinate
        axis (``freq_h, freq_w = freq[:, 0], freq[:, 1]`` for the (h, w) case; a
        third select for (t, h, w)), so ``N`` equals the number of distinct constant
        column indices selected from the frequency tensor — read straight off the
        source, no class-name or model-specific checks. Restricted to the
        vision-scoped class whose forward takes a ``position_ids`` argument (the
        rotary embedding), so a text rope is never matched. The forward line range
        lets callers recognise the op that reads the raw ``position_ids`` parameter
        (its ``@op_l<line>`` falls inside this range) and seed it authoritatively,
        since the inlined rotary frame has no ``@input`` boundary of its own.
        """
        best: tuple[int, int, int] | None = None
        for name, structure in self.spec.class_registry.items():
            if name not in self._vision_scoped:
                continue
            node = getattr(structure, "node", None)
            if not isinstance(node, ast.ClassDef):
                continue
            forward = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef) and item.name == "forward"
                ),
                None,
            )
            if forward is None:
                continue
            params = {arg.arg for arg in forward.args.args}
            if "position_ids" not in params:
                continue
            indices: set[int] = set()
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Subscript):
                    continue
                sl = sub.slice
                # A coordinate select ``freq[:, i]``: a 2-tuple subscript whose
                # first element is a full slice and second a constant int index.
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
                    first, second = sl.elts
                    if isinstance(first, ast.Slice):
                        value = _ast_const_int(second)
                        if value is not None:
                            indices.add(value)
            # Only trust a contiguous 0..N-1 coordinate fan (``freq[:, 0]``,
            # ``freq[:, 1]``); anything else is an unrelated subscript pattern.
            if len(indices) >= 2 and indices == set(range(len(indices))):
                end_line = getattr(forward, "end_lineno", None) or forward.lineno
                best = (len(indices), forward.lineno, end_line)
        return best

    def _descriptor_frame_input(
        self, node: ModelGraphNode, sources: list[str]
    ) -> list[TensorSpec] | None:
        """Input specs for an op reading a grid descriptor, or ``None``.

        An op inside a descriptor frame whose operand comes from OUTSIDE that
        frame is reading the frame's parameter -- the grid the model was handed.
        It has no producer in the graph, so the edge carries whatever the section
        was flowing (the flat image patches), and the whole chain then computes
        patch geometry. Provenance decides this, not shape: the patches and the
        tower's hidden activation are different shapes and either may be what the
        stale edge supplies.
        """
        if not self._descriptor_frames:
            return None
        frame = self._descriptor_frame_name(str(node.id))
        if frame is None:
            return None
        spec = self._descriptor_frames[frame]
        if not sources:
            return [spec]
        outside = [frame not in str(source) for source in sources]
        if not any(outside):
            return None
        return [
            spec if is_outside else self._tensor_specs.get(source, spec)
            for source, is_outside in zip(sources, outside)
        ]

    def _descriptor_frame_name(self, node_id: str) -> str | None:
        """The descriptor frame a node sits inside, longest match first."""
        best: str | None = None
        for attr_name in self._descriptor_frames:
            if attr_name in node_id and (best is None or len(attr_name) > len(best)):
                best = attr_name
        return best

    @staticmethod
    def _frame_prefix(node_id: str) -> str:
        """The frame a node belongs to: its id without the op segment."""
        head, sep, _ = str(node_id).rpartition(":@op_")
        return head if sep else ""

    def _record_frame_entry(self, node: Any, inputs: list[TensorSpec]) -> None:
        """Remember what a frame was handed, the first time it reads anything.

        Ops are inferred in forward order, so the first op of a frame reads the
        tensor the frame was given. ``setdefault`` keeps that one.
        """
        if not inputs or not inputs[0].shape:
            return
        prefix = self._frame_prefix(getattr(node, "id", ""))
        if prefix:
            self._frame_entry_shapes.setdefault(prefix, tuple(inputs[0].shape))

    def _frame_entry_shape(
        self, node: Any, owner: str | None
    ) -> tuple[str, tuple] | None:
        """``(parameter name, shape)`` of what this node's frame was handed.

        ``None`` unless the owning class names its forward input, since without
        that name there is nothing for a reshape to have referred to.
        """
        structure = (getattr(self.spec, "class_registry", None) or {}).get(owner or "")
        parameter = getattr(structure, "forward_input_name", None)
        if not parameter:
            return None
        shape = self._frame_entry_shapes.get(
            self._frame_prefix(getattr(node, "id", ""))
        )
        if shape is None:
            return None
        return str(parameter), shape

    def _entry_spec_for(
        self, node: ModelGraphNode, root: BlockNode | None
    ) -> TensorSpec | None:
        """Resolve a forward ``@input`` boundary to its true upstream shape.

        Precedence: an explicit per-graph override (``self._entry_specs`` keyed by
        the port label), then the vision patch-embed seed. Returns ``None`` so the
        caller applies the default language ``(B, S, H)`` seed — the text path and
        every non-vision model are therefore untouched.
        """
        override = self._entry_specs.get(node.label or "")
        if override is not None:
            return override
        # Seed the flat patch shape ``[Pv, C*T*P*P]`` for both the patch-embed
        # class (whose views/Conv3d then resolve concretely) and the vision tower
        # itself. The tower's forward flattens raw patches, calls the patch-embed,
        # then runs the blocks / post-norm / merger inline; those inner boundaries
        # inherit their shapes by conservation from this root ``@input``. Seeding
        # the tower with the language ``(B, S, H)`` default is what stamps the text
        # sequence axis ``B*S`` (and text hidden 4096) across the whole vision
        # tower — seeding ``[Pv, …]`` here flows the vision patch axis instead.
        seed_classes = {self._vision_patch_embed_class, self._vision_tower_class}
        if (
            self._vision_patch_flat is not None
            and root is not None
            and root.class_name in seed_classes
        ):
            return TensorSpec(
                shape=(Symbol.VISION_PATCH.value, self._vision_patch_flat),
                dtype=self.context.dtype,
            )
        # Meta-device ground truth for a forward parameter. Lowest precedence: it
        # only resolves a boundary the explicit override and vision seed both
        # declined, replacing the generic ``(B, S, hidden)`` default with the real
        # observed shape. The class-scoped map is consulted first — it resolves a
        # parameter whose name is globally ambiguous but locally definite (e.g.
        # ``attention_mask`` is ``[B, S]`` on the sparse-attention indexer but a
        # 4-D causal mask on the decoder attention). The global map only carries
        # parameters with no cross-module disagreement, so it never fires on an
        # ambiguous name (``hidden_states`` etc.).
        meta_input = self._meta_input_specs_for(node.label, root)
        if meta_input is not None:
            return meta_input
        # Nothing authoritative named this boundary; fall back to what the caller
        # hands this section, when that is known.
        return self._caller_entry_spec

    def _meta_input_specs_for(
        self, label: str | None, root: BlockNode | None
    ) -> TensorSpec | None:
        """Meta ground truth for a boundary ``label``, class-scoped first.

        Prefers the shape observed for this parameter *within the module class*
        that owns the boundary (``root.class_name``); falls back to the global
        cross-module-consistent shape. Returns *None* when neither is known.
        """
        param = label or ""
        self._ensure_meta_input_specs()
        if root is not None and root.class_name:
            scoped = self._meta_input_specs_by_class.get((root.class_name, param))
            if scoped is not None:
                return scoped
        return self._meta_input_specs.get(param)

    def boundary_input_spec(
        self,
        label: str | None,
        namespace: str | None = None,
        *,
        class_scoped_only: bool = False,
    ) -> TensorSpec | None:
        """Meta ground truth for a merged ``@input`` boundary, class-scoped first.

        The merge-time boundary tiles (``@input:<param>``) are not part of the
        ModelGraph that ``_entry_spec_for`` shapes, so ``fill_missing_node_shapes``
        seeds them from here instead of the generic ``(B, S, hidden)`` default.
        Resolution mirrors :meth:`_meta_input_specs_for`, but the owning module
        class is read from the boundary's ``namespace`` (its segments name the
        enclosing module classes) rather than a ``BlockNode`` root: try each
        namespace segment as a candidate class, then fall back to the global
        cross-module-consistent shape. Returns *None* when the parameter's shape
        was never observed on meta (so the caller keeps its existing heuristics).

        ``class_scoped_only`` suppresses the global fallback. A caller that
        *overrides* an already-resolved boundary (rather than filling a missing
        one) must trust only the locally-definite class-scoped shape: the global
        map is "consistent by absence" -- a parameter the meta trace happened to
        observe in only one module family (e.g. ``position_ids`` seen as ``(1, S)``
        on the text stack but never on the vision rotary, which uses ``[Pv, 2]``)
        lands in the global map and would wrongly clobber the other family's
        authoritative shape.
        """
        param = (label or "").strip()
        if not param:
            return None
        self._ensure_meta_input_specs()
        if namespace:
            for segment in reversed(str(namespace).split("/")):
                segment = segment.strip()
                if not segment:
                    continue
                scoped = self._meta_input_specs_by_class.get((segment, param))
                if scoped is not None:
                    return scoped
        if class_scoped_only:
            return None
        return self._meta_input_specs.get(param)

    def _ensure_meta_input_specs(self) -> None:
        """Populate the forward-parameter input specs on first use.

        ``load_meta_shapes`` fills these eagerly on the CLI path, but a direct
        ``build_merged_model_graph`` (the library / test path) never calls it.
        In that case trace them once, lazily, from the spec's checkpoint so
        ``boundary_input_spec`` sizes ``@input`` boundaries from meta ground
        truth in both paths. The result is cached (including the empty case) so
        the meta forward runs at most once per inferencer.
        """
        if self._meta_input_specs_loaded:
            return
        self._meta_input_specs_loaded = True
        checkpoint = self._meta_checkpoint_id()
        if not checkpoint:
            return
        try:
            from TraceLens.ModelUtils.meta_trace import trace_meta_input_specs

            input_specs = trace_meta_input_specs(
                checkpoint, config=self.spec.raw_config
            )
        except Exception:  # pragma: no cover - defensive; meta trace is best-effort
            return
        if not input_specs:
            return
        global_specs, class_specs = input_specs
        for param, (shape, dtype) in global_specs.items():
            self._meta_input_specs.setdefault(
                param, TensorSpec(shape=shape, dtype=dtype)
            )
        for (class_name, param), (shape, dtype) in class_specs.items():
            self._meta_input_specs_by_class.setdefault(
                (class_name, param), TensorSpec(shape=shape, dtype=dtype)
            )

    def _meta_checkpoint_id(self) -> str | None:
        """A checkpoint id ``AutoConfig.from_pretrained`` accepts, or *None*.

        The CLI path stores a bare, instantiable id on ``_meta_checkpoint``. The
        direct-build path only has ``spec.checkpoint_source``, which is a display
        label — for a Hub model it is ``hf://<owner>/<name>/<config-path>``, which
        ``AutoConfig`` rejects. Strip the scheme and keep the ``<owner>/<name>``
        repo id; pass any other source (local path, etc.) through unchanged.
        """
        explicit = self._meta_checkpoint
        if explicit:
            return str(explicit)
        source = str(getattr(self.spec, "checkpoint_source", "") or "").strip()
        if not source:
            return None
        if source.startswith("hf://"):
            parts = [seg for seg in source[len("hf://") :].split("/") if seg]
            if len(parts) >= 2:
                return "/".join(parts[:2])
            return "/".join(parts) or None
        return source

    def load_meta_shapes(
        self,
        checkpoint: str | Path,
        *,
        seq_len: int = 128,
        batch_size: int = 2,
    ) -> bool:
        """Run a meta-device forward pass and store per-module shapes.

        ``batch_size`` is 2, not 1, so :func:`symbolise_meta_shape` does not alias
        a genuine size-1 dim (an ``unsqueeze(1)`` head axis) onto ``B`` -- with
        ``batch_size == 1`` every singleton dim would print as ``B`` (a compressor
        output ``(1, 1, 32, 512)`` becoming a nonsensical ``[B, B, 32, 512]``).
        This mirrors the collision-free trace dims :func:`trace_meta_input_specs`
        already uses.

        Returns *True* when shapes were successfully captured.
        """
        from TraceLens.ModelUtils.meta_trace import (
            trace_meta_shapes,
            trace_meta_input_specs,
            symbolise_meta_shape,
        )

        # Retain for the lazy per-op FX fallback, even if module-level tracing
        # below captures nothing.
        self._meta_checkpoint = checkpoint

        # Globally-consistent forward-parameter input shapes (own collision-free
        # trace dims; see ``trace_meta_input_specs``). Seeds ``@input`` boundaries
        # that would otherwise take the generic activation default.
        input_specs = trace_meta_input_specs(checkpoint, config=self.spec.raw_config)
        # This eager fill supersedes the lazy ``_ensure_meta_input_specs`` trace.
        self._meta_input_specs_loaded = True
        if input_specs:
            global_specs, class_specs = input_specs
            for param, (shape, dtype) in global_specs.items():
                self._meta_input_specs[param] = TensorSpec(shape=shape, dtype=dtype)
            for (class_name, param), (shape, dtype) in class_specs.items():
                self._meta_input_specs_by_class[(class_name, param)] = TensorSpec(
                    shape=shape, dtype=dtype
                )

        raw = trace_meta_shapes(
            checkpoint,
            config=self.spec.raw_config,
            seq_len=seq_len,
            batch_size=batch_size,
        )
        if raw is not None:
            for module_path, shape in raw.items():
                sym = symbolise_meta_shape(
                    shape, batch_size=batch_size, seq_len=seq_len
                )
                self._meta_shapes[module_path] = TensorSpec(
                    shape=sym, dtype=self.context.dtype
                )
        return bool(
            self._meta_shapes
            or self._meta_input_specs
            or self._meta_input_specs_by_class
        )

    def infer_model_graph(
        self, graph: ModelGraph, *, root: BlockNode | None = None
    ) -> dict[str, TensorSpec]:
        """Infer output tensor specs for every node id in one model graph."""
        # Switch the default activation geometry to the vision patch axis while a
        # vision-scoped section is inferred; a root-less subgraph recursion keeps
        # whatever the enclosing section established. Restored before returning.
        prev_axes, prev_hidden = self._active_seq_axes, self._active_hidden
        if root is not None:
            if (
                root.class_name in self._vision_scoped
                and self._vision_hidden is not None
            ):
                self._active_seq_axes = (Symbol.VISION_PATCH.value,)
                self._active_hidden = self._vision_hidden
            else:
                self._active_seq_axes = (Symbol.BATCH.value, Symbol.SEQ.value)
                self._active_hidden = self.context.dims.get(
                    Symbol.HIDDEN.value, Symbol.HIDDEN.value
                )
        self._register_op_line_occurrences(graph)
        self._tensor_names = {}
        for node in graph.nodes:
            if node.metadata.get("synthetic") == "@input":
                self._tensor_names[node.id] = "input"
            else:
                self._tensor_names[node.id] = _output_tensor_name(node)
        self._tensor_specs = {}
        # Boundary tiles by the NAME they carry, so a reshape target naming
        # ANOTHER tensor resolves against that tensor rather than against
        # whatever is being reshaped.
        self._boundary_shapes = {}
        self._frame_entry_shapes = {}
        self._last_input_sources: list[str] = []
        # Multi-output nodes whose published slices genuinely divide the parent,
        # and so may be read as a consumer's operand (see
        # ``_publish_output_port_specs``). Reset with the specs they describe.
        self._tiling_slot_ids: set[str] = set()
        self._forward_input_specs = set()
        # Reset with the id set it guards; the owner-class refs outlive this,
        # since that cache does.
        self._forward_input_spec_refs = []
        self._entry_seeded_ids = set()
        self._rooted_output_ids = set()
        self._module_resolved_ids: set[str] = set()
        order = _topological_order(graph)
        node_by_id = {node.id: node for node in graph.nodes}
        # Consumers per source, for redocking the dedicated position_ids producer.
        consumers_of: dict[str, list[str]] = {}
        for edge in graph.edges:
            consumers_of.setdefault(edge.source, []).append(edge.target)

        for node_id in order:
            node = node_by_id[node_id]
            input_specs, input_labels = self._gather_input_specs(graph, node_id)
            self._record_frame_entry(node, input_specs)
            descriptor = self._descriptor_frame_input(node, self._last_input_sources)
            if descriptor is not None:
                input_specs = descriptor
                input_labels = [""] * len(descriptor)
            seeded = self._vision_position_ids_input(node)
            if seeded is not None:
                input_specs = seeded
                input_labels = [""] * len(seeded)
                # The rotary op reads position_ids from a dedicated host producer
                # (``get_vision_position_ids``); its host index-bookkeeping cannot
                # be shape-inferred, so it carries a bogus shape that the merged
                # ``@input:position_ids`` boundary would otherwise inherit. Stamp
                # that producer (the source feeding only this op) with the same
                # authoritative ``[Pv, N]`` seed so the boundary reads correctly.
                for edge in graph.edges:
                    if edge.target != node_id:
                        continue
                    if consumers_of.get(edge.source) == [node_id]:
                        self._tensor_specs[edge.source] = seeded[0]
            output = self._infer_node_output(
                node, input_specs, root=root, input_labels=input_labels
            )
            output = self._resolve_extent_dims(node, output, input_specs)
            output = _with_explicit_dtype(node, output)
            if root is not None and node.metadata.get("synthetic") == "@output":
                self._rooted_output_ids.add(node_id)
                if node_id == "@output" and getattr(root, "class_name", None):
                    self._module_output_specs.setdefault(
                        str(root.class_name), set()
                    ).add((tuple(output.shape), output.dtype))
            self._tensor_specs[node_id] = output
            if node.metadata.get("synthetic") == "@input" and node.label:
                self._boundary_shapes.setdefault(str(node.label), output.shape)
            # Publish this node's per-ordinal slices NOW, not in a pass after the
            # whole graph is inferred: a consumer reading one slot of a split is
            # itself a later node in this same loop, and until the slice exists it
            # can only fall back to the undivided tensor. That is how a
            # ``cat((q_pass, q_rot))`` of a [..., 128] and a [..., 64] slice came
            # out [..., 384] -- it was handed the [..., 192] parent twice.
            self._publish_output_port_specs(node, self._tensor_specs)
            if node.metadata.get("synthetic") == "@input":
                self._forward_input_specs.add(id(output))
                self._forward_input_spec_refs.append(output)
                if self._entry_spec_for(node, root) is not None:
                    self._entry_seeded_ids.add(node_id)

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

        # Snapshot this graph's own specs before recursing: the recursion below
        # re-initializes shared instance state (``self._tensor_specs`` etc.), so
        # without capturing them here every top-level spec would be clobbered and
        # only the final subgraph's specs would survive. Merge each subgraph's
        # inferred specs (their node ids are globally unique) into the result so
        # callers see the full graph, not just the last subgraph.
        merged = dict(self._tensor_specs)
        # Node ids this (parent) graph authoritatively seeded from context. The
        # subgraph recursion below runs root-less, so its @input boundaries fall
        # back to the default activation spec; where a boundary id collides with a
        # parent-seeded one (e.g. the vision patch-embed @input = [Pv, C*T*P*P]),
        # the parent's in-context spec must win rather than be clobbered by the
        # root-less default (which would stamp the generic [Pv, hidden]).
        # Also protect nodes this graph sized from a module's constructor
        # dimensions: that lookup needs the class context ``root`` supplies,
        # and the recursion below deliberately runs without one, so its
        # class-less result for the same id is strictly less informed.
        # An ``@output`` resolved in class context is the module's real return.
        # Without the class the recursion resolves the same id against whatever
        # the body's last op happened to be: DeepSeek's indexer returns its int64
        # ``[B, S, 6]`` top-k picks, and the root-less pass reported the scorer's
        # float32 ``[B, S, 64, 32]`` scores in their place -- which every consumer
        # of the indexer then inherited.
        seeded = (
            set(self._entry_seeded_ids)
            | set(self._module_resolved_ids)
            | set(self._rooted_output_ids)
        )
        for node in graph.nodes:
            if node.kind == NodeKind.SUBGRAPH:
                subgraph_key = node.metadata.get("subgraph_key")
                if subgraph_key and subgraph_key in graph.subgraphs:
                    sub_specs = self.infer_model_graph(graph.subgraphs[subgraph_key])
                    for spec_id, spec in sub_specs.items():
                        if spec_id in seeded and spec_id in merged:
                            continue
                        merged[spec_id] = spec

        # A loop's entry port reports what the body hands back, not what seeded
        # it. They are the same variable: from the second iteration on, the body
        # consumes its own output, so the body's shape is the one the loop
        # actually runs on. The seed is sized from its producer, and a producer
        # whose only rule is to echo its input -- a module with no shape rule of
        # its own -- would otherwise make the whole loop report the width from
        # BEFORE that module ran (GLM's vision loop carried the raw patch width
        # [Pv, C*T*P*P] instead of the embedded [Pv, hidden]).
        for node in graph.nodes:
            if node.metadata.get("synthetic") != "@loop_carried":
                continue
            carried_back = [
                edge.source
                for edge in graph.edges
                if edge.target == node.id
                and (source := node_by_id.get(edge.source)) is not None
                and source.metadata.get("synthetic") == "@loop_carried"
            ]
            if len(carried_back) == 1 and merged.get(carried_back[0]) is not None:
                merged[node.id] = merged[carried_back[0]]

        # Per-output-port slices for tuple-unpacked split/chunk/unbind. The node's
        # own spec is the whole (pre-split) tensor; each consumer edge selects an
        # ordinal. Publish one spec per ordinal under a reserved key so the
        # exporter can label each output port with its slice shape ([B, S, 16] for
        # ``comb_w``) instead of the undivided tensor.
        for node in graph.nodes:
            self._publish_output_port_specs(node, merged)

        self._tensor_specs = merged
        self._active_seq_axes, self._active_hidden = prev_axes, prev_hidden
        return dict(merged)

    def infer_block_tree(
        self,
        root: BlockNode,
        *,
        title: str = "",
        entry_spec: TensorSpec | None = None,
    ) -> dict[str, TensorSpec]:
        """Build a model graph from a block tree and infer all node shapes.

        ``entry_spec`` is what the CALLER hands this tree. A method expansion is
        inferred as its own section with no caller context, so without it the
        ``@input`` falls back to the activation default and the whole body
        computes the enclosing module's geometry.
        """
        from TraceLens.ModelUtils.basic_ops import BasicOpFilter

        basic_ops = self.spec.basic_ops or BasicOpFilter.for_detailed()
        graph = build_model_graph(root, title=title or root.label, basic_ops=basic_ops)
        previous = self._caller_entry_spec
        self._caller_entry_spec = entry_spec
        try:
            return self.infer_model_graph(graph, root=root)
        finally:
            self._caller_entry_spec = previous

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
        # Fill size-1 axes of the widest operand from concrete axes another
        # operand supplies (``[Pv, ?, 1] * [16] -> [Pv, ?, 16]`` in axial RoPE).
        # Only genuine 1->n broadcasts change the result; every other case keeps
        # the previous inputs[0] passthrough.
        if widest.shape and any(dim == 1 for dim in widest.shape):
            axes = list(widest.shape)
            changed = False
            for other in inputs:
                if other is widest or not other.shape:
                    continue
                for offset in range(1, len(axes) + 1):
                    if offset > len(other.shape):
                        break
                    other_dim = other.shape[-offset]
                    if (
                        axes[-offset] == 1
                        and isinstance(other_dim, int)
                        and other_dim != 1
                    ):
                        axes[-offset] = other_dim
                        changed = True
            if changed:
                return TensorSpec(shape=tuple(axes), dtype=widest.dtype)
        return inputs[0]

    def _spec_through_port(
        self, graph: ModelGraph, node_id: str, _depth: int = 0
    ) -> TensorSpec | None:
        """The spec of whatever feeds *node_id*, for a pass-through port.

        Only for a node that carries a value rather than computing one, and only
        with exactly one producer, so nothing is invented about an op that
        genuinely combines several inputs.
        """
        if _depth > 4:
            return None
        node = next((item for item in graph.nodes if item.id == node_id), None)
        if node is None:
            return None
        if str((node.metadata or {}).get("synthetic") or "") not in _PASS_THROUGH_PORTS:
            return None
        sources = [edge.source for edge in graph.edges if edge.target == node_id]
        if len(sources) != 1:
            return None
        spec = self._tensor_specs.get(sources[0])
        if spec is not None:
            return spec
        return self._spec_through_port(graph, sources[0], _depth + 1)

    def _slot_spec(self, edge: GraphEdge, occurrence: int) -> TensorSpec | None:
        """The spec of the output SLOT *edge* reads, else the producer's own.

        An edge out of a multi-output op names the ordinal it takes
        (``source_port``), either as a single value or -- when one consumer reads
        several slots of the same producer -- as a list consumed in edge order.
        Reading the producer's undivided spec instead silently hands a consumer
        the parent tensor: a ``cat`` of two slices then sums the parent's width
        once per slice.
        """
        port = edge.source_port
        ordinal: str | None = None
        if isinstance(port, (list, tuple)):
            if occurrence < len(port):
                ordinal = str(port[occurrence])
        elif port is not None:
            ordinal = str(port)
        if ordinal is not None and edge.source in self._tiling_slot_ids:
            sliced = self._tensor_specs.get(f"{edge.source}{PORT_SPEC_SEP}{ordinal}")
            if sliced is not None:
                return sliced
        return self._tensor_specs.get(edge.source)

    def _publish_output_port_specs(
        self, node: ModelGraphNode, specs: dict[str, TensorSpec]
    ) -> None:
        """Record one spec per ordinal for a tuple-unpacked split/chunk/unbind.

        The node's own spec is the whole (pre-split) tensor; each consumer edge
        selects an ordinal. Publishing each slice under a reserved key lets both
        the exporter label the output port with its slice shape and a consumer
        read the slot it actually takes.
        """
        output_names = node.metadata.get("output_names")
        if not output_names or len(output_names) < 2:
            return
        class_name = (node.metadata.get("class_name") or node.label or "").strip()
        op_label = (node.label or class_name).strip().lower()
        if op_label not in {"split", "chunk", "unbind"}:
            return
        whole = specs.get(node.id)
        if whole is None:
            return
        details = [str(item) for item in node.metadata.get("details", [])]
        slices: list[TensorSpec | None] = []
        for ordinal in range(len(output_names)):
            sliced = _multi_output_slice_shape(
                whole,
                details,
                op_label,
                ordinal,
                self.context.dims,
                output_count=len(output_names),
            )
            slices.append(sliced)
            if sliced is not None:
                specs[f"{node.id}{PORT_SPEC_SEP}{ordinal}"] = sliced
        # A slice is good enough to LABEL an output port -- it is the best thing
        # we can say about that port -- without being good enough to compute
        # with. Only a set of slices that genuinely divides the parent may feed a
        # consumer's shape rule: GLM's indexer has a 3-way split of a [B, S, 257]
        # whose recorded sizes read 1/0/0, and handing a consumer a zero-width
        # operand silently drops an axis for the rest of the chain. Where the
        # division does not add up, consumers stay on the undivided tensor.
        resolved = [item for item in slices if item is not None]
        if len(resolved) == len(slices) and _slices_tile(
            whole, resolved, removes_axis=op_label == "unbind"
        ):
            self._tiling_slot_ids.add(node.id)

    def _gather_input_specs(
        self, graph: ModelGraph, node_id: str
    ) -> tuple[list[TensorSpec], list[str]]:
        """Ordered input specs feeding ``node_id`` and their port labels.

        The label is the port role a producer feeds through -- read from a
        ``@kernel_in:<idx>:<label>`` port node's id (the attention kernel names its
        query/key/value/mask ports there) and otherwise from the edge's own label.
        It lets a shape rule that needs to tell operands apart (the ``sdpa`` output
        rule reads *query* and *value* by role) do so without trusting raw edge
        order. Parallel to the spec list: index ``i``'s label describes spec ``i``.
        """
        specs: list[TensorSpec] = []
        labels: list[str] = []
        sources: list[str] = []
        node_by_id = {item.id: item for item in graph.nodes}
        # One consumer can read two different slots of the same producer
        # (``torch.cat((q_pass, q_rot))`` where both come from one ``split``).
        # That is carried as two parallel edges sharing one ordinal LIST, taken in
        # order -- so count the occurrences of each source to know which slot this
        # edge is.
        occurrences: dict[str, int] = {}
        for edge in graph.edges:
            if edge.target != node_id:
                continue
            occurrence = occurrences.get(edge.source, 0)
            occurrences[edge.source] = occurrence + 1
            source_spec = self._slot_spec(edge, occurrence)
            if source_spec is None:
                # A port CARRIES a value rather than computing one. If it has no
                # spec of its own yet, read through to its producer rather than
                # dropping the operand: an op left with NO inputs falls back to
                # the section's activation shape, which is how GLM's vision
                # ``sdpa`` reported a 2-D ``[Pv, 1024]`` while each of its three
                # ports was a correct ``[1, 16, Pv, 64]``.
                source_spec = self._spec_through_port(graph, edge.source)
            if source_spec is None:
                continue
            sources.append(edge.source)
            specs.append(source_spec)
            # Ask the PORT which operand it supplies. It was stamped when the
            # port was bound, so a port renamed for the reader still answers.
            # Only a port that never declared one falls back to its id.
            source_node = node_by_id.get(edge.source)
            role = (
                str((source_node.metadata or {}).get("operand_role") or "")
                if source_node is not None
                else ""
            )
            if role:
                labels.append(role)
            elif "@kernel_in:" in edge.source:
                labels.append(edge.source.rsplit(":", 1)[-1])
            else:
                labels.append(edge.label or "")
        # The producers behind these operands, for a rule that must tell whether
        # a ``X.shape[i]`` target names the tensor being reshaped or another one.
        self._last_input_sources = sources
        return specs, labels

    def _vision_position_ids_input(
        self, node: ModelGraphNode
    ) -> list[TensorSpec] | None:
        """Authoritative ``[Pv, N]`` input for the vision rotary ``position_ids`` op.

        The vision rotary embedding is inlined into the tower forward, so its
        ``position_ids`` parameter has no ``@input`` boundary to seed. Instead we
        seed the single internal op that first reads that parameter — identified,
        name-agnostically, by its ``position_ids`` port and by its source line
        falling inside the rotary forward (see ``_vision_position_ids_seed``). Its
        incoming edges otherwise carry the flat patch geometry (``[Pv, 1176]``) or
        the host index-bookkeeping producer's mis-inferred shape; both are wrong.
        Returns ``None`` (leave edge specs untouched) for every other node.
        """
        seed = self._vision_posid_seed
        if seed is None:
            return None
        # Only while a vision-scoped section is the active geometry.
        if self._active_seq_axes != (Symbol.VISION_PATCH.value,):
            return None
        if (node.metadata.get("port_label") or "") != "position_ids":
            return None
        spec, start_line, end_line = seed
        line = _op_line_of(node.id)
        if line is None or not (start_line <= line <= end_line):
            return None
        return [spec]

    def _resolve_extent_dims(
        self,
        node: ModelGraphNode,
        output: TensorSpec,
        input_specs: list[TensorSpec],
    ) -> TensorSpec:
        """Replace a ``<tensor>.shape[i]`` dim with the extent it actually names.

        A rule that cannot evaluate a recorded bound reports the expression, so
        ``torch.arange(key_states.shape[2])`` reports ``[key_states.shape[2]]``
        -- the source line rather than the length -- and every op downstream
        inherits it. The extractor marks the operands wired to such an op purely
        for their EXTENT, and the named tensor is one of them, so read the axis
        off those operands. Only when they all agree on it: two extent operands
        disagreeing means the expression could name either, and a guess would be
        worse than the expression.
        """
        dims = [str(dim) for dim in output.shape]
        if not any(".shape[" in dim for dim in dims):
            return output
        details = list(node.metadata.get("details") or ())
        raw_count = _detail_value(details, "extent_inputs")
        try:
            count = int(str(raw_count).strip())
        except (TypeError, ValueError):
            return output
        operands = input_specs[-count:] if count else []
        if not operands:
            return output
        resolved: list[Any] = []
        changed = False
        for dim in dims:
            match = re.fullmatch(r".+\.shape\[(-?\d+)\]", dim)
            if match is not None:
                index = int(match.group(1))
                candidates = {
                    operand.shape[index]
                    for operand in operands
                    if -len(operand.shape) <= index < len(operand.shape)
                }
                if len(candidates) == 1:
                    resolved.append(candidates.pop())
                    changed = True
                    continue
            resolved.append(dim)
        if not changed:
            return output
        return TensorSpec(shape=tuple(resolved), dtype=output.dtype)

    def _infer_node_output(
        self,
        node: ModelGraphNode,
        inputs: list[TensorSpec],
        *,
        root: BlockNode | None,
        input_labels: list[str] | None = None,
    ) -> TensorSpec:
        dtype = self.context.dtype
        synthetic = node.metadata.get("synthetic")
        class_name = (node.metadata.get("class_name") or node.label or "").strip()
        block_class = class_name or node.label

        # A module drawn as a subgraph reports what its own graph returns. It is
        # sized here before that graph is walked, so the fallback passes its input
        # through -- which is how the indexer, handed hidden states and returning
        # int64 picks, reported hidden states to everything reading it.
        if node.kind == NodeKind.SUBGRAPH and class_name:
            returned = self._module_output_specs.get(class_name)
            if returned and len(returned) == 1:
                shape, returned_dtype = next(iter(returned))
                return TensorSpec(shape=shape, dtype=returned_dtype or dtype)

        if synthetic == "@input":
            if (node.label or "").lower() in {"input_ids", "input"}:
                return TensorSpec(
                    shape=(Symbol.BATCH.value, Symbol.SEQ.value), dtype="int64"
                )
            entry = self._entry_spec_for(node, root)
            if entry is not None:
                return entry
            return self._activation_spec(dtype)

        if synthetic in {"@output", "@loop_carried"}:
            if inputs:
                return inputs[-1]
            return self._activation_spec(dtype)

        if node.metadata.get("constant") and not inputs:
            # A materialized constant/buffer leaf (e.g. the rotary ``inv_freq``):
            # its output *is* the parameter/buffer tensor, so size it directly from
            # the meta parameter/buffer registry rather than any incoming edge.
            names = [str(name) for name in node.metadata.get("external_inputs", [])]
            if names:
                spec = self.constant_spec(node, root=root, names=names)
                if spec is not None:
                    return spec
                # A ``self.<attr>`` read that resolves to no registered
                # parameter/buffer is a scalar hyper-parameter (an ``eps``, a
                # ``scaling`` factor), not an activation. Size it as a single
                # element ``[1]`` rather than falling through to the ``(B, S, H)``
                # activation default (which would be a wrong operand shape in the
                # constants view and trip the "no shape rule" warning). A 0-d
                # ``()`` cannot be used: the display formatter drops an empty
                # shape, so the node would be re-filled with the section default.
                return TensorSpec(shape=(1,), dtype=self.context.dtype)

        if synthetic == "@tensor":
            label = (node.metadata.get("port_label") or node.label or "").lower()
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            hidden = self.context.dims.get(Symbol.HIDDEN.value, Symbol.HIDDEN.value)
            parameter = self._lookup_parameter_spec(node, root=root, names=[label])
            if parameter is not None:
                # A buffer whose ``__init__`` states its dtype keeps it: a routing
                # table is int64, and reading it as an activation loses the
                # integer-ness every index consumer depends on.
                param_dtype = parameter.dtype or (
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
            operands = list(inputs)
            # A hidden buffer operand (axial-RoPE ``inv_freq``, folded onto this
            # op by ``_tag_buffer_only_ops`` so the buffer stays invisible)
            # carries a captured shape the visible edges lost. Fold it back in so
            # broadcasting recovers the real width (``[Pv, ?, 1] * inv_freq[16]``).
            for name in node.metadata.get("external_inputs", []):
                parameter = self._lookup_parameter_spec(
                    node, root=root, names=[str(name)]
                )
                if parameter is not None and parameter.shape:
                    operands.append(TensorSpec(shape=parameter.shape, dtype=dtype))
            if operands:
                if node.label in {"+", "Add"}:
                    widest = max(operands, key=_broadcast_rank)
                    # Rank alone picks the operand with the most axes, which can
                    # still carry a placeholder 1 in the axis a sibling actually
                    # sizes: ``tail_start[..., None] + tail_offsets`` is [B, 1, 1]
                    # plus [max_tail_width] and must keep the offsets' width, or
                    # the concat consuming it reports a single-element tail.
                    # Deliberately only a trailing literal 1, and only when one
                    # sibling width is in play, so rank, dtype and every other
                    # axis are untouched.
                    shape = widest.shape
                    if shape and shape[-1] == 1:
                        widths = {
                            item.shape[-1]
                            for item in operands
                            if item.shape and item.shape[-1] != 1
                        }
                        if len(widths) == 1:
                            return TensorSpec(
                                shape=(*shape[:-1], next(iter(widths))),
                                dtype=widest.dtype,
                            )
                    return widest
                return self._elementwise_operand(operands)
            return self._activation_spec(dtype)

        # Catch-all for any remaining synthetic wiring nodes (kernel ports,
        # hidden_states, etc.) — silent passthrough, no warning.
        if synthetic is not None and synthetic.startswith("@"):
            # An ``@input_mirror`` shows the value flowing in from the enclosing
            # scope; its true shape is the boundary parameter's. When that
            # parameter is a globally-consistent meta ground truth (e.g.
            # ``attention_mask`` → ``[B, S]``), prefer it over the passthrough,
            # whose upstream would otherwise be the generic activation default.
            if synthetic == "@input_mirror":
                meta_input = self._meta_input_specs_for(node.label, root)
                if meta_input is not None:
                    return meta_input
            if inputs:
                return inputs[0]
            return self._activation_spec(dtype)

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

        if operation_label == "arange":
            # ``torch.arange`` fabricates a 1-D range from host-scalar bounds; it
            # reads no tensor operand, so it must size its own axis rather than
            # inherit a neighbour's shape. ``_arange_axis`` folds all-integer
            # bounds to a concrete length and otherwise keeps the runtime length
            # (``compressed_len``/``n_windows``) as a symbolic extent.
            owner = self._owner_class_name(node, root=root) or (
                root.class_name if root is not None else None
            )
            owner_scalars = (
                self.module_dims.scalar_by_class.get(owner) if owner else None
            )
            arange_dims = self.context.dims
            if owner_scalars:
                arange_dims = {**self.context.dims, **owner_scalars}
            length = _arange_axis(details, arange_dims)
            return TensorSpec(shape=(length,), dtype="int64")

        if operation_label in {
            "ones",
            "zeros",
            "empty",
            "full",
            # ``x.new_empty(*sizes)`` and kin: the same thing spelled as a
            # method, with the receiver supplying only dtype and device.
            "new empty",
            "new zeros",
            "new ones",
            "new full",
        }:
            # Built from host scalars, so it must size its own axes rather than
            # inherit a neighbour's: ``torch.ones(B, S, dtype=torch.bool)``
            # reads no tensor operand at all. Each size resolves the way an
            # ``arange`` bound does -- an int literal, a known dim, or the bare
            # identifier kept as a symbolic extent.
            owner = self._owner_class_name(node, root=root) or (
                root.class_name if root is not None else None
            )
            owner_scalars = (
                self.module_dims.scalar_by_class.get(owner) if owner else None
            )
            size_dims = self.context.dims
            if owner_scalars:
                size_dims = {**self.context.dims, **owner_scalars}
            whole = _detail_value(details, "size0")
            if (
                whole is not None
                and _detail_value(details, "size1") is None
                and whole.strip().endswith(".shape")
                and inputs
            ):
                # ``torch.zeros(keep.shape, ...)`` is sized BY another tensor,
                # not by a list of scalars: the one edge wired here carries that
                # tensor's extent, so its shape is the answer.
                return TensorSpec(
                    shape=inputs[0].shape,
                    dtype=_constructed_dtype(details) or inputs[0].dtype,
                )
            # A ``new_*`` constructor's receiver is its TEMPLATE: it supplies
            # dtype and device, and the sizes may name its leading axes
            # (``kv_nope.new_empty(*kv_nope.shape[:-1], head_dim)``).
            template = inputs[0] if inputs else None
            shape = _constructed_shape(details, size_dims, template)
            if shape is None:
                return inputs[0] if inputs else self._activation_spec(dtype)
            return TensorSpec(
                shape=shape,
                dtype=_constructed_dtype(details)
                or (template.dtype if template is not None else dtype),
            )

        if operation_label == "flatten":
            # Collapse axes ``[start_dim, end_dim]`` (inclusive) into one, per
            # ``torch.flatten``. Defaults: start_dim=0, end_dim=-1. Unlike
            # view/reshape there is no target-shape detail, so the old code path
            # (shared with view) always passed the tensor through unchanged,
            # leaving phantom rank downstream.
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            shape = source.shape
            if not shape:
                return source
            start_detail = _detail_value(details, "start_dim")
            end_detail = _detail_value(details, "end_dim")
            if start_detail is None and end_detail is None:
                # No captured span. A bare ``flatten()`` collapses everything,
                # but a missing detail would otherwise be indistinguishable
                # from "args not captured", so only the call site saying it
                # took NO span lets us collapse; anything else passes through
                # rather than guess.
                if _detail_value(details, "flatten_all") is None:
                    return source
                collapsed = _merge_axes(shape)
                if collapsed is None:
                    return source
                return TensorSpec(shape=(collapsed,), dtype=source.dtype)
            rank = len(shape)
            start = _int_dim(start_detail)
            end = _int_dim(end_detail)
            start = 0 if start is None else start % rank
            end = rank - 1 if end is None else end % rank
            if start >= end or not (0 <= start < rank and 0 <= end < rank):
                # Single-axis or unresolvable span: nothing to merge.
                return source
            merged = _merge_axes(shape[start : end + 1])
            if merged is None:
                return source
            return TensorSpec(
                shape=shape[:start] + (merged,) + shape[end + 1 :],
                dtype=source.dtype,
            )

        if operation_label == "pad":
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            raw = _detail_value(details, "pad")
            if not raw or not source.shape:
                return source
            amounts: list[int] = []
            for token in raw.split(","):
                token = token.strip()
                if not token.lstrip("-").isdigit():
                    return source
                amounts.append(int(token))
            pairs = len(amounts) // 2
            if len(amounts) % 2 or pairs > len(source.shape):
                return source
            padded = list(source.shape)
            for index in range(pairs):
                added = amounts[2 * index] + amounts[2 * index + 1]
                if not added:
                    continue
                axis = len(padded) - 1 - index
                padded[axis] = _sum_dim_sizes([padded[axis], added])
            return TensorSpec(shape=tuple(padded), dtype=source.dtype)

        if operation_label in {"rearrange", "repeat"}:
            # einops already states the answer; apply it rather than passing
            # the tensor through at a rank the pattern just changed.
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            pattern = _detail_value(details, "pattern")
            if not pattern or not source.shape:
                return source
            owner = self._owner_class_name(node, root=root) or (
                root.class_name if root is not None else None
            )
            axis_dims = dict(self.context.dims)
            owner_scalars = (
                self.module_dims.scalar_by_class.get(owner) if owner else None
            )
            if owner_scalars:
                axis_dims.update(owner_scalars)
            sizes: dict[str, DimExpr] = {}
            for item in details:
                if not item.startswith("axis "):
                    continue
                name, _, raw = item[len("axis ") :].partition(":")
                token = raw.strip()
                resolved: DimExpr | None = (
                    int(token)
                    if token.lstrip("-").isdigit()
                    else _resolve_dim_name(token, axis_dims)
                )
                if resolved is not None:
                    sizes[name.strip()] = resolved
            rearranged = _einops_shape(pattern, source.shape, sizes)
            if rearranged is None:
                return source
            return TensorSpec(shape=rearranged, dtype=source.dtype)

        if operation_label in {"view", "reshape"}:
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
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
            # A reshape can name the FRAME's own input rather than the tensor it
            # is reshaping. GPT-2's `Conv1D` flattens to `[B*S, in]`, projects,
            # then restores `x.size()[:-1] + (nf,)` -- the leading axes of the
            # `x` it was handed, which is `[B, S, in]`. Reading them from the
            # source instead gives `[B*S, nf]`: self-consistent, one axis short,
            # and the `split(..., dim=2)` that follows then has no axis 2.
            # Scoped to this frame, because `x` names a different tensor in every
            # other one.
            frame_shapes = dict(self._boundary_shapes)
            entry = self._frame_entry_shape(node, owner)
            if entry is not None:
                frame_shapes[entry[0]] = entry[1]
            resolved = _resolve_view_shape(
                shape_detail,
                source,
                view_dims,
                frame_shapes,
                _shape_snapshot_tokens(details),
            )
            if resolved is not None:
                return TensorSpec(shape=resolved, dtype=source.dtype)
            if "-1" in shape_detail:
                flattened = self._active_flattened_seq()
                return TensorSpec(
                    shape=(flattened, source.shape[-1]), dtype=source.dtype
                )
            # A view STATES its own rank, so returning the source contradicts the
            # call. A dimension this scope cannot resolve to a number is still a
            # dimension: ``pool_offsets.view(1, number_of_pools, self.index_kpool)``
            # stayed rank 1, and the advanced index reading it, the ``flatten(-2)``
            # after that and the concat at the end all inherited the missing axes.
            # Keep the stated rank, carrying an unresolved name as a symbolic
            # extent -- a starred prefix (``*x.shape[:-1]``) stands for an unknown
            # NUMBER of axes, so that one genuinely cannot be counted.
            tokens = [token.strip() for token in shape_detail.split(",")]
            if tokens and all(tokens) and not any("*" in token for token in tokens):
                dims: list[Any] = []
                for token in tokens:
                    number = _as_int(token)
                    if number is not None:
                        dims.append(number)
                        continue
                    # ``self.block_size`` is this module's own scalar.
                    resolved_dim = view_dims.get(token)
                    if resolved_dim is None and token.startswith("self."):
                        resolved_dim = view_dims.get(token.removeprefix("self."))
                    if resolved_dim is not None:
                        dims.append(resolved_dim)
                        continue
                    # A plain name stands for an extent this scope cannot put a
                    # number to but can still carry (``number_of_pools``). Anything
                    # that is still an EXPRESSION (``hidden_states.shape[0]``) is
                    # not a dimension name, and printing it as one states a shape
                    # nobody can read -- keep the old reading there.
                    if not _PLAIN_DIM_NAME.fullmatch(token):
                        return source
                    dims.append(token)
                return TensorSpec(shape=tuple(dims), dtype=source.dtype)
            return source

        if operation_label == "unsqueeze":
            source = inputs[0] if inputs else external_spec() or TensorSpec((), dtype)
            dim_str = _detail_value(details, "dim")
            dim = _int_dim(dim_str) if dim_str is not None else 0
            if dim is None:
                dim = 0
            rank = len(source.shape)
            insert_at = dim if dim >= 0 else rank + 1 + dim
            insert_at = max(0, min(insert_at, rank))
            return TensorSpec(
                shape=(*source.shape[:insert_at], 1, *source.shape[insert_at:]),
                dtype=source.dtype,
            )

        if operation_label == "slice":
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            select_str = _detail_value(details, "select_dim")
            drop: set[int] = set()
            if select_str:
                for token in select_str.split(","):
                    parsed = _int_dim(token.strip())
                    if parsed is not None and source.shape:
                        drop.add(parsed % len(source.shape))
            if drop:
                return TensorSpec(
                    shape=tuple(
                        size
                        for axis, size in enumerate(source.shape)
                        if axis not in drop
                    ),
                    dtype=source.dtype,
                )
            # A strided range-slice (``x[..., 0::2]``) thins the axis rather than
            # bounding it: interleaved RoPE takes every other element, so the axis
            # holds ceil((dim - start) / step). A symbolic axis is left alone --
            # the stride is only applied to a width we can actually divide.
            step_str = _detail_value(details, "step_dim")
            if step_str and source.shape:
                shape = list(source.shape)
                rank = len(shape)
                for token in step_str.split(","):
                    axis_str, _, spec_str = token.strip().partition("=")
                    start_str, _, stride_str = spec_str.partition(":")
                    axis = _int_dim(axis_str.strip())
                    start = _int_dim(start_str.strip())
                    stride = _int_dim(stride_str.strip())
                    if axis is None or start is None or not stride or stride <= 1:
                        continue
                    dim = _int_dim(shape[axis % rank])
                    if dim is None:
                        continue
                    shape[axis % rank] = max(0, -(-(dim - start) // stride))
                return TensorSpec(shape=tuple(shape), dtype=source.dtype)
            # A bounded range-slice to a config-derived constant
            # (``topk_indices[..., :output_width]``) resizes that axis to the
            # folded width, overriding whatever the source axis held (it may have
            # been inflated by proven data-dependent internals upstream).
            resize_str = _detail_value(details, "resize_dim")
            if resize_str and source.shape:
                shape = list(source.shape)
                rank = len(shape)
                for token in resize_str.split(","):
                    axis_str, _, size_str = token.strip().partition("=")
                    axis = _int_dim(axis_str.strip())
                    try:
                        size = int(size_str.strip())
                    except (TypeError, ValueError):
                        size = None
                    if axis is not None and size is not None:
                        shape[axis % rank] = size
                return TensorSpec(shape=tuple(shape), dtype=source.dtype)
            # A range slice whose bound is arithmetic over the operand's OWN shape
            # (``rotate_half``'s ``x[..., : x.shape[-1] // 2]``) is sized here: the
            # per-axis ``lower|upper`` expressions reference a ``shape`` symbol we bind
            # to this operand's concrete shape. An axis whose bound cannot be evaluated
            # to an int (a symbolic dim) is left unchanged.
            shape_slice_str = _detail_value(details, "shape_slice")
            if shape_slice_str and source.shape:
                shape = list(source.shape)
                rank = len(shape)
                for token in shape_slice_str.split(", "):
                    axis_str, _, bounds = token.strip().partition("=")
                    axis = _int_dim(axis_str.strip())
                    if axis is None:
                        continue
                    lower_s, _, upper_s = bounds.partition("|")
                    raw_dim = shape[axis % rank]
                    dim = _int_dim(raw_dim)
                    lo = _eval_shape_expr(lower_s.strip(), shape)
                    hi = _eval_shape_expr(upper_s.strip(), shape)
                    if (
                        dim is not None
                        and (lo is None or isinstance(lo, int))
                        and (hi is None or isinstance(hi, int))
                    ):
                        # Every bound resolved to a concrete int: fold the slice to
                        # an exact numeric width, same as before this dim could be
                        # symbolic.
                        if lo is None:
                            lo = 0
                        if hi is None:
                            hi = dim
                        if lo < 0:
                            lo += dim
                        if hi < 0:
                            hi += dim
                        shape[axis % rank] = max(0, min(hi, dim) - max(lo, 0))
                        continue
                    # The axis itself or one of its bounds is symbolic (a named
                    # dim like ``'S'`` rather than a numeric literal): a floor-div
                    # or other narrowing can't be folded to an int, but the
                    # narrowed axis must still end up DISTINCT from the
                    # unmodified source dim (a rotate_half-style half-width slice
                    # must not report output_shape == input_shape). Render a
                    # symbolic expression string instead.
                    if hi is not None and lo is None:
                        shape[axis % rank] = hi
                    elif lo is not None and hi is None:
                        shape[axis % rank] = f"{raw_dim}-({lo})"
                    elif lo is not None and hi is not None:
                        shape[axis % rank] = f"({hi})-({lo})"
                    # else: neither bound resolved (e.g. a no-op full slice) --
                    # leave this axis unchanged.
                return TensorSpec(shape=tuple(shape), dtype=source.dtype)
            # A range slice bounded by a LOCAL name -- partial RoPE's
            # ``rotary_dim = cos.shape[-1]`` then ``q[..., :rotary_dim]`` and
            # ``q[..., rotary_dim:]`` -- resolves to no integer, so both halves
            # used to report the FULL width. The concat that rejoins them then
            # reported double: MiniMax's query reached ``sdpa`` 256 wide against
            # a 128-wide key, a product that cannot be formed. The two are
            # complementary by construction, so narrow them symbolically and let
            # summing them cancel back to the source width.
            bound = _slice_bound_name(_detail_value(details, "slice"))
            if bound is not None and source.shape:
                name, takes_head = bound
                width = source.shape[-1]
                narrowed = name if takes_head else f"{width}-({name})"
                return TensorSpec(
                    shape=source.shape[:-1] + (narrowed,), dtype=source.dtype
                )
            return source

        if operation_label == "merge":
            # An explicit branch-merge (phi): exactly one of its mutually-exclusive
            # inputs flows through per invocation, and both carry the same shape,
            # so the output is just that shape -- a pure passthrough of any operand.
            # This is NOT ``torch.select``, which indexes one position along a
            # dimension and so REMOVES that dimension from the output shape.
            return (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )

        if operation_label == "tile":
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            repeat_str = _detail_value(details, "repeat")
            dim_str = _detail_value(details, "dim")
            dim = _int_dim(dim_str) if dim_str is not None else -1
            if dim is None:
                dim = -1
            try:
                repeat = int(repeat_str) if repeat_str is not None else 1
            except (TypeError, ValueError):
                repeat = 1
            if source.shape and repeat > 1:
                resolved_dim = dim % len(source.shape)
                dim_val = source.shape[resolved_dim]
                if isinstance(dim_val, int):
                    return TensorSpec(
                        shape=_replace_dim(
                            source.shape, resolved_dim, dim_val * repeat
                        ),
                        dtype=source.dtype,
                    )
            return source

        if operation_label in {"split", "chunk", "unbind"}:
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            # A tuple-unpacked split exposes one output port per slice (sized by
            # the ``PORT_SPEC_SEP`` post-pass in ``infer_model_graph``); the split
            # node itself represents the whole tensor being divided, so its own
            # spec passes the source through.
            if node.metadata.get("output_names"):
                return source
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
                return TensorSpec(self._active_hidden_shape(), dtype)
            if operation_label == "stack":
                # ``torch.stack`` inserts a brand-new size-N axis at ``dim``
                # (default 0) -- NOT always at the front. Blindly prepending it
                # regardless of the recorded ``dim`` detail (e.g. DeepSeek's
                # ``rotate_half``: ``torch.stack((-x2, x1), dim=-1)``) plants
                # the phantom axis at the wrong position; a later
                # ``.flatten(-2)`` merging "the last two axes" then merges the
                # WRONG pair, leaving the phantom axis stranded at the front
                # instead of absorbed.
                base = inputs[0]
                dim_str = _detail_value(details, "dim")
                dim = _int_dim(dim_str) if dim_str is not None else 0
                if dim is None:
                    dim = 0
                rank_out = len(base.shape) + 1
                pos = dim % rank_out if rank_out else 0
                shape = base.shape[:pos] + (len(inputs),) + base.shape[pos:]
                return TensorSpec(shape=shape, dtype=base.dtype)
            # Concat: the output matches every input except along the concat
            # axis, whose size is the SUM of the inputs' sizes there. torch.cat
            # requires all inputs to share a rank, so a negative dim names the same
            # axis from the end for each -- resolve the axis PER OPERAND so a
            # (mis-inferred) rank mismatch can't silently drop the shorter operand
            # and fabricate an identity concat (output == one input's shape). Sum
            # every operand's contribution at that axis -- concrete sizes fold
            # into one running total, symbolic sizes (e.g. "S*8") join it as
            # terms -- so a single symbolic contributor no longer forces a bail
            # out to "keep the widest operand's shape" (which silently swallows
            # every other operand's width, including concrete ones).
            dim_str = _detail_value(details, "dim")
            if dim_str is None:
                # ``torch.cat([a, b])`` joins along axis 0, the same default
                # ``stack`` takes just above -- assuming the TRAILING axis is the
                # one case where leaving the argument out changed the answer.
                dim = 0
            else:
                # A ``dim`` that is named but not a literal (``cat(parts, dim=d)``)
                # leaves the axis unknown; the trailing axis remains the better
                # guess there than axis 0.
                dim = _int_dim(dim_str)
                if dim is None:
                    dim = -1
            base = max(inputs, key=_broadcast_rank)
            base_dim = dim % len(base.shape) if base.shape else 0
            concat_sizes = []
            usable = True
            for inp in inputs:
                if not inp.shape:
                    usable = False
                    break
                inp_dim = dim % len(inp.shape) if dim < 0 else dim
                if inp_dim < 0 or inp_dim >= len(inp.shape):
                    usable = False
                    break
                concat_sizes.append(inp.shape[inp_dim])
            if usable and concat_sizes:
                total = _sum_dim_sizes(concat_sizes)
                return TensorSpec(
                    shape=_replace_dim(base.shape, base_dim, total),
                    dtype=base.dtype,
                )
            return base

        if operation_label in {"transpose", "permute"}:
            source = (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
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
            return TensorSpec(self._active_hidden_shape(), dtype)

        # Scaled-dot-product attention's compiled core: the output keeps the
        # query's leading axes ``[..., S_q]`` and takes its last dim from the value
        # head dim, ``out = query.shape[:-1] + (value.shape[-1],)``. This is
        # shape-correct for grouped-query attention (repeat_kv expands the query
        # head *count*, not value's head dim) and for latent attention where
        # ``v_head_dim != qk_head_dim`` (value's own last dim carries it). Query and
        # value are told apart by the ROLE each kernel port declared when it was
        # bound, not by operand order or by the op's name, so a swapped edge
        # order never mis-picks and renaming a port for the reader cannot break
        # it. The rule fires for any op handed a ``query`` and a ``value`` --
        # which is what makes something attention-shaped -- rather than for ops
        # we recognise by name.
        roles = [str(label).lower() for label in (input_labels or [])]
        if "query" in roles and "value" in roles:
            query_spec: TensorSpec | None = None
            value_spec: TensorSpec | None = None
            for spec, role in zip(inputs, roles):
                if role == "query" and query_spec is None:
                    query_spec = spec
                if role == "value" and value_spec is None:
                    value_spec = spec
            if (
                query_spec is not None
                and value_spec is not None
                and query_spec.shape
                and value_spec.shape
            ):
                out_shape = (*query_spec.shape[:-1], value_spec.shape[-1])
                return TensorSpec(shape=out_shape, dtype=query_spec.dtype)
            if inputs:
                return inputs[0]
            return TensorSpec(self._active_hidden_shape(), dtype)

        if operation_label == "einsum":
            equation = _detail_value(details, "equation")
            if equation and "->" in equation and inputs:
                out_shape = _infer_einsum_shape(equation, inputs)
                if out_shape is not None:
                    return TensorSpec(shape=out_shape, dtype=inputs[0].dtype)
            if inputs:
                return inputs[0]
            return TensorSpec(self._active_hidden_shape(), dtype)

        if operation_label == "nonzero":
            source = (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
            )
            ndim = len(source.shape) if source.shape else 1
            return TensorSpec(shape=("nnz", ndim), dtype="int64")

        if operation_label == "one hot":
            source = (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
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
            return TensorSpec(self._active_hidden_shape(), dtype)

        if operation_label in {"cast", "contiguous"}:
            # Both keep the source shape. ``contiguous`` is a pure layout no-op
            # (shape *and* dtype pass through); ``cast`` reads its ``dtype:``
            # detail to change dtype only.
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            cast_dtype = source.dtype
            if operation_label == "cast":
                dtype_detail = _detail_value(details, "dtype") or ""
                if dtype_detail:
                    cast_dtype = _resolve_cast_dtype(
                        dtype_detail, source.dtype, self.context.dtype
                    )
            return TensorSpec(shape=source.shape, dtype=cast_dtype)

        if operation_label == "squeeze":
            # Drop a size-1 axis (torch no-ops on any other size). ``x.squeeze(dim)``
            # normalizes a negative ``dim`` against rank; bare ``x.squeeze()`` drops
            # every size-1 axis.
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            shape = source.shape
            if not shape:
                return source
            dim_str = _detail_value(details, "dim")
            if dim_str is None:
                # Bare squeeze: drop all size-1 axes.
                reduced = tuple(size for size in shape if size != 1)
                return TensorSpec(shape=reduced, dtype=source.dtype)
            dim = _int_dim(dim_str)
            if dim is None:
                return source
            axis = dim % len(shape)
            if 0 <= axis < len(shape) and shape[axis] == 1:
                return TensorSpec(
                    shape=tuple(
                        size for index, size in enumerate(shape) if index != axis
                    ),
                    dtype=source.dtype,
                )
            return source

        if operation_label == "expand":
            # Broadcast a size-1 axis to a concrete target. Uses its OWN resolver:
            # unlike view/reshape, ``expand``'s ``-1`` means "keep this axis" (not
            # element conservation). Falls back to passthrough when the target
            # cannot be positionally aligned, never corrupting rank.
            source = (
                inputs[0]
                if inputs
                else external_spec() or TensorSpec(self._active_hidden_shape(), dtype)
            )
            shape_detail = _detail_value(details, "shape") or ""
            resolved = _resolve_expand_shape(shape_detail, source, self.context.dims)
            if resolved is not None:
                return TensorSpec(shape=resolved, dtype=source.dtype)
            return source

        if operation_label == "topk":
            source = (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
            )
            top_k = self.context.dims.get(
                Symbol.EXPERTS_PER_TOK.value, Symbol.EXPERTS_PER_TOK.value
            )
            # Honour this call's own k when it is a concrete literal. The
            # experts-per-token default is right for the gate's final expert pick
            # but wrong for a group-scoring ``.topk(2, dim=-1)`` on the way there,
            # which would otherwise report the expert width. A non-literal k (a
            # config-derived attr) keeps the default rather than being guessed at.
            k_detail = _detail_value(details, "k")
            if k_detail is not None:
                text = str(k_detail).strip()
                if text.isdigit() and int(text) > 0:
                    top_k = int(text)
            # ``k`` narrows the axis ``dim`` names. That axis is the trailing one
            # by default, but a gate scoring expert GROUPS takes its top few along
            # another (``scores.topk(2, dim=1)``), and narrowing the last axis
            # instead reports a width that tensor never had.
            axes = _reduction_axes(_detail_value(details, "dim"))
            rank = len(source.shape)
            if axes and rank:
                axis = axes[0]
                if -rank <= axis < rank:
                    dims = list(source.shape)
                    dims[axis] = top_k
                    return TensorSpec(shape=tuple(dims), dtype="int64")
            return TensorSpec(
                shape=_replace_last_dim(source.shape, top_k), dtype="int64"
            )

        if operation_label == "index select":
            # ``base[index]`` advanced indexing whose index operand is a tensor
            # parameter (``index_first_axis(x, indices): return x[indices]``). The
            # integer index operand(s) replace the *leading* axes of ``base`` and
            # the trailing base axes are kept: ``out = broadcast(index shapes) ++
            # base.shape[num_index_operands:]``. A dedicated label -- never the
            # overloaded ``gather`` branch below, which single-index/no-``dim`` uses
            # for many non-row-gather shapes. When the index has no resolved integer
            # shape yet (an upstream still-opaque producer), pass the base shape
            # through rather than inventing one; never emit a warning.
            int_indices = [item for item in inputs if item.dtype == "int64"]
            base = next(
                (item for item in inputs if item.dtype not in {"int64", "bool"}), None
            )
            if base is not None and int_indices:
                idx_shape = _broadcast_shapes([item.shape for item in int_indices])
                tail = tuple(base.shape[len(int_indices) :])
                return TensorSpec(shape=tuple(idx_shape) + tail, dtype=base.dtype)
            if base is not None:
                return base
            return (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
            )

        if operation_label == "gather":
            has_dim = _detail_value(details, "dim") is not None
            # A buffer this op reads is the TABLE being indexed, whatever its
            # dtype. Separating base from index by "the non-integer one" works
            # only while the table is floating-point: a token -> expert routing
            # table is int64 exactly like the ids that index it, and then that
            # rule finds no base at all and the lookup collapses to the index's
            # own shape. The op names its buffer, so ask it.
            table_shapes: set[tuple[DimExpr, ...]] = set()
            for name in node.metadata.get("external_inputs", []) or ():
                buffer_spec = self._lookup_parameter_spec(
                    node, root=root, names=[str(name)]
                )
                if buffer_spec is not None and buffer_spec.shape:
                    table_shapes.add(tuple(buffer_spec.shape))
            base = next(
                (item for item in inputs if tuple(item.shape) in table_shapes), None
            )
            if base is None:
                base = next(
                    (item for item in inputs if item.dtype not in {"int64", "bool"}),
                    None,
                )
            int_indices = [
                item for item in inputs if item is not base and item.dtype == "int64"
            ]
            # Integer advanced indexing ``base[i, j, ...]`` is a ``Subscript`` /
            # ``__getitem__`` and -- unlike ``torch.gather`` -- carries no ``dim``
            # detail. With two or more integer index operands it indexes the
            # *leading* axes: the broadcast of the index operands replaces those
            # axes and the trailing base axes are kept. ``torch.gather`` (always a
            # single index operand with a ``dim``) and boolean-mask indexing are
            # left on the path below, so every real ``torch.gather`` is unchanged.
            # Indexing the LEADING axes: the broadcast of the index operands
            # replaces them and the trailing base axes are kept. Two cases reach
            # it -- several index operands, or a single one reading a BUFFER,
            # which is a lookup table by construction. The ``gather`` label is
            # overloaded (single-index/no-``dim`` covers many non-row-gather
            # shapes across these models), so a lone index into an ordinary
            # activation deliberately keeps the legacy ``index.shape`` answer
            # rather than being reinterpreted on a guess.
            if (
                not has_dim
                and base is not None
                and (
                    len(int_indices) >= 2
                    or (int_indices and tuple(base.shape) in table_shapes)
                )
            ):
                idx_shape = _broadcast_shapes([item.shape for item in int_indices])
                tail = tuple(base.shape[len(int_indices) :])
                return TensorSpec(shape=tuple(idx_shape) + tail, dtype=base.dtype)
            source = base if base is not None else (inputs[0] if inputs else None)
            index = int_indices[0] if int_indices else None
            if source is None:
                source = TensorSpec(self._active_hidden_shape(), dtype)
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
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
            )
            index_reduction = operation_label in {"argmax", "argmin"}
            boolean_reduction = operation_label in {"any", "all"}
            out_dtype = _reduction_dtype(
                source.dtype, index_reduction, boolean_reduction
            )
            # torch reductions default to ``keepdim=False`` -- a reduced axis is
            # dropped, not collapsed to size 1. Only an explicit ``keepdim=True``
            # keeps it. Collapsing unconditionally (an earlier behaviour)
            # fabricates a phantom axis a downstream ``cat``/``stack`` then
            # disagrees with its sibling operand's real rank on.
            keepdim = _detail_value(details, "keepdim") == "True"
            axes = _reduction_axes(_detail_value(details, "dim"))
            if axes is not None:
                # Every axis the call names is reduced, whether it names one
                # (``sum(dim=2)``) or several (``expert_mask.sum(dim=(-1, -2))``,
                # which every MoE router runs). Reducing only the first, or none,
                # publishes the tensor from BEFORE the reduction -- the scorer's
                # boundary reported ``[B, S, H, T]`` where its docstring says
                # ``[B, S, T]``, and the router's ``greater``/``nonzero`` read two
                # axes that were already summed away.
                rank = len(source.shape)
                present = sorted(
                    {axis + rank if axis < 0 else axis for axis in axes}
                    & set(range(rank))
                )
                if present:
                    dims = list(source.shape)
                    if keepdim:
                        for axis in present:
                            dims[axis] = 1
                    else:
                        for axis in reversed(present):
                            del dims[axis]
                    return TensorSpec(shape=tuple(dims), dtype=out_dtype)
                # An axis the symbolic (B, S, H) view omits -- a stream or head
                # axis this shape does not carry -- cannot be dropped from it.
                if boolean_reduction:
                    return TensorSpec(shape=source.shape, dtype="bool")
                if not index_reduction:
                    return source
                return TensorSpec(shape=source.shape, dtype="int64")
            # No axis named at all (``dim`` absent, or spelled ``None``): read it
            # as the trailing axis, which is what every such call in the four
            # models means.
            if keepdim:
                return TensorSpec(
                    shape=_replace_last_dim(source.shape, 1), dtype=out_dtype
                )
            return TensorSpec(shape=source.shape[:-1], dtype=out_dtype)

        if operation_label in _COMPARISON_LABELS:
            # Element-wise comparison: broadcast to the widest operand, boolean out.
            if inputs:
                source = max(inputs, key=_broadcast_rank)
                return TensorSpec(shape=source.shape, dtype="bool")
            return TensorSpec(shape=self._active_hidden_shape(), dtype="bool")

        # An activation module resolved from a registry (e.g. ``act_fn = ACT2FN[...]``)
        # keeps its attribute name as the label (``act_fn``) while its class is the
        # concrete activation (``SiLU``); match on the class so it is treated as the
        # pointwise op it is rather than falling through to the shape warning.
        if (
            operation_label in _POINTWISE_LABELS
            or (class_name or "").strip().lower() in _POINTWISE_LABELS
        ):
            if inputs:
                if operation_label in _FIRST_OPERAND_WRITE_LABELS:
                    # ``Tensor.masked_fill(mask, value)`` and its sibling in-place
                    # writes (``masked_scatter``/``scatter``/``scatter_add``/
                    # ``index_add``/``copy_``) return a tensor shaped and typed like
                    # operand 0 -- the tensor being written into. The trailing
                    # mask/index/source operands broadcast or address into it but
                    # never define the result, so anchor on operand 0 rather than
                    # letting a wider bool mask win the ``_broadcast_rank`` vote.
                    return TensorSpec(shape=inputs[0].shape, dtype=inputs[0].dtype)
                source = max(inputs, key=_broadcast_rank)
                if operation_label == "where":
                    # ``torch.where(condition, x, y)`` takes its VALUES from x and
                    # y; the condition only selects between them. Letting the
                    # condition win the dtype vote turns a selected index into a
                    # mask -- DeepSeek's ``torch.where(valid, top_k_indices, ...)``
                    # reported bool, so the scatter reading it had no integer
                    # operand at all. The shape still broadcasts across all three.
                    values = [item for item in inputs if item.dtype != "bool"]
                    if values:
                        return TensorSpec(
                            shape=source.shape,
                            dtype=max(values, key=_broadcast_rank).dtype,
                        )
                return TensorSpec(shape=source.shape, dtype=source.dtype)
            return TensorSpec(shape=self._active_hidden_shape(), dtype=dtype)

        linear_spec = self._lookup_linear_spec(node, root=root)
        if linear_spec is not None:
            # Resolved from the owning module's own constructor dimensions,
            # which needs the class context ``root`` carries. Remember it so a
            # later root-less subgraph pass cannot overwrite it with a
            # class-less guess.
            self._module_resolved_ids.add(node.id)
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
                else self._active_hidden_shape()
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
            if out_features is None and _detail_value(details, "raw_op") == "addmm":
                # `addmm(bias, x, weight)` is an affine projection that stores
                # its weight [in, out] -- the TRANSPOSE of `F.linear`'s
                # [out, in] -- so the output width is the column axis, not the
                # row one. That is how every GPT-2 `Conv1D` projects, and both
                # of its learned operands reach the op as constant leaves whose
                # shapes are not resolved yet, so the weight is read the way a
                # materialized constant is.
                weight_names = [
                    name for name in external_inputs if "weight" in str(name).lower()
                ]
                weight = self.constant_spec(
                    node, root=root, names=weight_names or external_inputs
                )
                if weight is not None and len(weight.shape) >= 2:
                    out_features = weight.shape[-1]
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
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
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
            return self._activation_spec(dtype)

        if node.operation == OperationKind.GPU_KERNEL or class_name in {
            "AttentionOp",
            "KernelOp",
            "KernelOutput",
            "AttentionMerge",
        }:
            introspected = self._introspect_forward_shape(node, inputs, root=root)
            if introspected is not None:
                return introspected
            return self._activation_spec(dtype)

        if _is_router(block_class, node):
            experts = self.context.dims.get(Symbol.EXPERTS.value, Symbol.EXPERTS.value)
            in_shape = inputs[0].shape if inputs else self._active_hidden_shape()
            return TensorSpec(shape=_replace_last_dim(in_shape, experts), dtype=dtype)

        if node.operation == OperationKind.TORCH_FUNCTIONAL:
            introspected = self._introspect_forward_shape(node, inputs, root=root)
            if introspected is not None:
                return introspected
            fx_spec = self._fx_op_shape(node, inputs)
            if fx_spec is not None:
                return fx_spec
            torch_op_spec = self._torch_op_shape(node, inputs)
            if torch_op_spec is not None:
                return torch_op_spec
            triton_spec = self._triton_op_shape(node, inputs)
            if triton_spec is not None:
                return triton_spec
            preserved = self._shape_preserving_torch_op(node, inputs)
            if preserved is not None:
                return preserved
            _log.warning(
                "No shape inference rule for %s (label=%r, class=%r); "
                "passing through input shape",
                node.id,
                node.label,
                node.metadata.get("class_name"),
            )
            if inputs:
                return inputs[0]
            return self._activation_spec(dtype)

        if (
            node.kind in {NodeKind.BLOCK, NodeKind.TOP_LEVEL}
            or node.operation == OperationKind.COMPOSITE
        ):
            if inputs:
                return inputs[0]
            return self._activation_spec(dtype)

        introspected = self._introspect_forward_shape(node, inputs, root=root)
        if introspected is not None:
            return introspected

        # Fallback: attention modules produce (B, S, H) — derive from config.
        node_name = (
            node.metadata.get("attr_name") or node.id.rsplit(":", 1)[-1] or ""
        ).lower()
        if "attention" in node_name:
            return self._activation_spec(dtype)

        # A custom submodule whose forward no symbolic rule or AST simulation
        # resolved (``compressor`` -- data-dependent windowing, tuple return):
        # take its real output shape from the meta-traced module instead of
        # passing the input shape through (which drops the rank/last-dim change).
        if node.operation == OperationKind.NN_MODULE:
            meta_module_spec = self._lookup_meta_module_shape(node)
            if meta_module_spec is not None:
                return meta_module_spec

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
        return TensorSpec(shape=self._active_hidden_shape(), dtype=dtype)

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

    def _shape_preserving_torch_op(
        self, node: ModelGraphNode, inputs: list[TensorSpec]
    ) -> TensorSpec | None:
        """Ask torch whether an op keeps its input's shape, using stand-in sizes.

        ``nn.Dropout`` has no symbolic rule and cannot be meta-executed on
        ``[B, S, 768]``, because ``B`` and ``S`` are not numbers. But the only
        question that matters is whether the output has the SAME shape as the
        input, and substituting a size for each symbol answers it -- then the
        real symbolic shape is carried through unchanged.

        This is what the fallback already did; the difference is that torch has
        now said so, rather than the shape being passed through with a warning
        because nothing knew. Distinct stand-ins per symbol, so an op that
        permutes or collapses axes cannot look shape-preserving by coincidence.
        One tensor operand only: with several there is no single input shape for
        the output to have preserved.
        """
        if len(inputs) != 1 or not inputs[0].shape:
            return None
        try:
            import torch
        except ImportError:
            return None
        stand_ins: dict[Any, int] = {}
        sizes: list[int] = []
        for dim in inputs[0].shape:
            concrete = self._concrete_dim(dim)
            if concrete is not None and concrete > 0:
                sizes.append(int(concrete))
                continue
            # Distinct and >1, so a transpose or a flatten cannot come back
            # looking like the shape it started with.
            sizes.append(stand_ins.setdefault(dim, 2 + 2 * len(stand_ins) + 1))
        try:
            meta = torch.zeros(
                sizes, dtype=_torch_dtype(torch, inputs[0].dtype), device="meta"
            )
        except Exception:  # noqa: BLE001
            return None
        details = [str(item) for item in node.metadata.get("details", [])]
        scalar_args = _op_scalar_detail_args(details, self.context.dims)
        for name in self._op_callable_candidates(node):
            fn = _resolve_meta_op_callable(torch, name)
            if fn is None:
                continue
            out = _run_meta_op(torch, fn, name, [meta], scalar_args)
            if out is None:
                # The op's own schema says what else it takes. `aten::dropout`
                # is `(Tensor input, float p, bool train)` with no defaults, so
                # a bare call cannot run it. Neutral values are sound HERE and
                # only here: the answer being read off is whether the shape
                # changed, which no probability or training flag decides.
                out = _run_meta_op(
                    torch, fn, name, [meta], _neutral_scalar_args(torch, name)
                )
            if out is None:
                continue
            shape = _first_tensor_shape(out)
            if shape is not None and tuple(shape) == tuple(sizes):
                return inputs[0]
        return None

    def _symbolise_concrete(self, shape: tuple[int, ...]) -> tuple[Any, ...]:
        from TraceLens.ModelUtils.meta_trace import symbolise_meta_shape

        return symbolise_meta_shape(
            shape, batch_size=_TORCH_PROBE_BATCH, seq_len=_TORCH_PROBE_SEQ
        )

    def _op_callable_candidates(self, node: ModelGraphNode) -> list[str]:
        """Op names to try resolving to a real callable, most-authoritative first.

        The ``raw_op`` recorded from the model's own forward source is preferred
        (``div``/``cumsum``/``split``/...); the op's display label is a weak
        fallback for nodes that carry no ``raw_op``. Neither is a curated op set
        -- both are names the model itself produced, resolved by attribute lookup.
        """
        details = [str(item) for item in node.metadata.get("details", [])]
        candidates: list[str] = []
        raw = _detail_value(details, "raw_op")
        if raw:
            candidates.append(raw)
        # A stage decomposed from a Triton kernel records the operation it
        # performs; its display label is a glyph that resolves to nothing.
        triton_op = _detail_value(details, "triton_op")
        if triton_op and triton_op not in candidates:
            candidates.append(triton_op)
        label = self._torch_op_label(node)
        if label and label not in candidates:
            candidates.append(label)
        return candidates

    def _triton_op_shape(
        self, node: ModelGraphNode, inputs: list[TensorSpec]
    ) -> TensorSpec | None:
        """Output shape of a stage decomposed from a Triton kernel.

        For these two families the Triton language fixes the result shape
        regardless of how the call was written, so it resolves without running
        anything: elementwise ops broadcast their operands, and the associative
        scans return their input unchanged. The torch fallbacks cannot help
        here -- ``tl.cumsum``'s axis is a kernel-launch detail we never
        recorded, so a meta execution of ``torch.cumsum`` has no ``dim`` to
        pass and simply fails.

        Returns *None* for anything else (a reduction, say) so the caller keeps
        looking rather than inventing a shape.
        """
        details = [str(item) for item in node.metadata.get("details", [])]
        triton_op = _detail_value(details, "triton_op")
        if not triton_op or not inputs:
            return None
        if triton_op in _TRITON_SCAN_OPS:
            return inputs[0]
        if triton_op in _TRITON_ELEMENTWISE_OPS:
            shape = _broadcast_shapes([item.shape for item in inputs])
            return TensorSpec(shape=shape, dtype=inputs[0].dtype)
        return None

    def _torch_op_shape(
        self, node: ModelGraphNode, inputs: list[TensorSpec]
    ) -> TensorSpec | None:
        """Run the op on the meta device to read its true output shape, when the
        symbolic arithmetic can't resolve it (e.g. a split whose size is a
        config-derived name, or an op with no symbolic rule at all).

        Generic: the op's real callable is resolved by attribute lookup from the
        name the model itself calls (``raw_op`` / display label) across the torch
        namespaces + aten registry -- there is no per-op builder allowlist. The
        callable is invoked on meta tensors built from the operand shapes/dtypes,
        with scalar arguments recovered from the recorded call details. Best-effort:
        returns None on any gap (unmapped dim, unresolvable callable, data-dependent
        or custom op that can't run on meta) so the caller passes through, never
        fabricating a shape a real meta execution wouldn't produce."""
        if not inputs:
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
        scalar_args = _op_scalar_detail_args(details, self.context.dims)
        for name in self._op_callable_candidates(node):
            fn = _resolve_meta_op_callable(torch, name)
            if fn is None:
                continue
            out = _run_meta_op(torch, fn, name, metas, scalar_args)
            if out is None:
                continue
            shape = _first_tensor_shape(out)
            if shape is None:
                continue
            return TensorSpec(self._symbolise_concrete(shape), inputs[0].dtype)
        return None

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

    def _op_line_key(self, node: ModelGraphNode) -> tuple[int, str, int] | None:
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

    def _lookup_meta_shape(self, node: ModelGraphNode) -> TensorSpec | None:
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

    def _lookup_meta_module_shape(self, node: ModelGraphNode) -> TensorSpec | None:
        """Last-resort shape for a submodule whose ``forward`` no symbolic rule or
        AST simulation could resolve, taken from the real module's meta-traced
        output.

        The merged graph collapses every repeated layer onto one representative
        node, so its id carries only the leaf attribute (``compressor``) with no
        layer index -- while ``_meta_shapes`` is keyed by full
        ``named_modules()`` paths (``model.layers.2.self_attn.compressor``). Match
        every meta path whose trailing attribute segments equal the node's own
        attribute chain, and (repeated layers differ only in data-dependent dims
        like a window count) return the single shape they agree on, or the modal
        one. Purely structural: keyed on the node's attribute name against real
        module paths, no per-model logic.
        """
        if not self._meta_shapes:
            return None
        attr = (
            _node_attr_name(node) or node.id.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        ).strip()
        if not attr:
            return None
        matches = [
            spec
            for path, spec in self._meta_shapes.items()
            if path.rsplit(".", 1)[-1] == attr
        ]
        if not matches:
            return None
        counter: Counter[tuple[Any, ...]] = Counter(spec.shape for spec in matches)
        best_shape, _count = counter.most_common(1)[0]
        for spec in matches:
            if spec.shape == best_shape:
                return spec
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
            # Derive output from config: attention preserves the activation
            # geometry of its section (``(B, S, H)`` text, ``(Pv, Hv)`` vision).
            if "attention" in (structure.name or "").lower():
                return TensorSpec(
                    shape=self._active_hidden_shape(),
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
        inline_result = self._introspect_inline_function(class_name, inputs, root=root)
        if inline_result is not None:
            return inline_result

        # class_name may be a method name (e.g. "get_pooled_states") rather
        # than a class.  Search the registry for a class that owns this method
        # and parse the method body for shape-bearing operations.
        method_result = self._introspect_method_shape(class_name, inputs, root=root)
        if method_result is not None:
            return method_result

        # For GPU kernels, try resolving the kernel source.
        details = [str(d) for d in node.metadata.get("details", [])]
        kernel_import = parse_kernel_import(details)
        if kernel_import is not None:
            return self._introspect_kernel_source(kernel_import, inputs, root=root)
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
            func_node = self._find_init_inline_function(structure.node, attr_name)
            if func_node is None:
                continue
            # Parse the inline function body using the same approach as
            # _introspect_method_shape.
            try:
                input_name = func_node.args.args[0].arg if func_node.args.args else "x"
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
            input_spec = (
                inputs[0] if inputs else TensorSpec(self._active_hidden_shape(), dtype)
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

    def _module_class_candidates(
        self, node: ModelGraphNode, root: BlockNode | None
    ) -> list[str]:
        """Classes that could own ``node``'s module, most specific first.

        Node metadata is authoritative when present; otherwise the owning
        submodule class (read off the id path) then the block's own class.
        Used to scope ``(class, attr)`` lookups so an attr name shared across
        classes (e.g. ``proj`` = Conv in a patch embed, Linear in a merger) is
        never resolved against the wrong class.
        """
        known = self.module_dims.known_classes()
        candidates: list[str] = []

        def _add(name: str | None) -> None:
            # Only real module-owning classes scope the lookup; a primitive op
            # label (``Conv2d``/``View``) in node metadata is not an owner, so it
            # must not block the attr-only fallback for an unscoped node.
            if name and name in known and name not in candidates:
                candidates.append(name)

        _add(node.metadata.get("class_name"))
        # Every module segment along the id path is a potential declaring class
        # for this node's attr, most specific first. A leaf module call (``qkv``)
        # resolves its own attr segment to its instance class (``Linear``, a
        # primitive that is filtered out here); the enclosing module
        # (``Glm5NextVisionAttention``) is the class that DECLARES ``self.qkv`` and
        # owns the ``(class, attr)`` linear/conv spec. Walk the whole ancestor
        # chain — not just the first match — so an inline-expanded leaf resolves
        # against its declaring class even when node metadata carries no class.
        if root is not None:
            classes = self._owner_classes.get(id(root))
            if classes is None:
                classes = _descendant_classes(root)
                self._owner_classes[id(root)] = classes
                self._owner_class_refs.append(root)
            for segment in reversed(re.split(r"[:/]", node.id)):
                if segment.startswith("@"):
                    continue
                _add(classes.get(segment))
        _add(root.class_name if root is not None else None)
        return candidates

    def _lookup_linear_spec(
        self, node: ModelGraphNode, *, root: BlockNode | None
    ) -> ModuleLinearSpec | None:
        attr = _node_attr_name(node)
        if not attr:
            return None
        candidates = self._module_class_candidates(node, root)
        # An instantiation-specific spec wins over the per-class one: the same
        # class built with different constructor arguments has different widths,
        # and only the owner (class + the attr it was assigned to) says which.
        if self.module_dims.linear_by_owner:
            for segment in re.split(r"[:/]", str(node.id)):
                if not segment or segment.startswith("@") or segment == attr:
                    continue
                key = (segment, attr)
                if key in self.module_dims.linear_owner_ambiguous:
                    continue
                spec = self.module_dims.linear_by_owner.get(key)
                if spec is not None:
                    return spec
        for class_name in candidates:
            spec = self.module_dims.linear.get((class_name, attr))
            if spec is not None:
                return spec
        # Only fall back to the (cross-class) attr-only index when the owning
        # class is genuinely unknown; a known class that lacks this attr means
        # the node is not that module's linear.
        if not candidates:
            return self.module_dims.linear_by_attr.get(attr)
        return None

    def _lookup_conv_spec(
        self, node: ModelGraphNode, *, root: BlockNode | None
    ) -> ModuleConvSpec | None:
        attr = _node_attr_name(node)
        if not attr:
            return None
        candidates = self._module_class_candidates(node, root)
        for class_name in candidates:
            spec = self.module_dims.conv.get((class_name, attr))
            if spec is not None:
                return spec
        if not candidates:
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
            self._owner_class_refs.append(root)
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
            bare = bare[len(prefix) :]
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
) -> DimExpr | None:
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
        if not remaining:
            # A purely numeric flatten dim must be a real ``int`` (``1``, ``64``),
            # not the string ``"1"``/``"64"``. Downstream size checks compare
            # ``dim == 1`` (squeeze dropping a size-1 axis, expand broadcasting a
            # size-1 axis, broadcast-rank); a stringified ``"1"`` fails those and
            # leaks a phantom "symbolic" axis (e.g. ``squeeze(2)`` refusing to
            # collapse ``[B, S, "1", 128]``).
            return num
        factors = remaining + ([str(num)] if num != 1 else [])
        return "*".join(factors)
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


def _parse_einops_side(side: str) -> list[Any] | None:
    """Axis terms of one side of an einops pattern, or None if unsupported.

    A term is an axis name, ``"..."`` for the ellipsis, or a tuple of names for
    a parenthesised group. ``1`` is accepted as a literal singleton axis.
    """
    terms: list[Any] = []
    index = 0
    while index < len(side):
        char = side[index]
        if char.isspace():
            index += 1
            continue
        if char == "(":
            close = side.find(")", index)
            if close == -1:
                return None
            inner = side[index + 1 : close].split()
            if any(not _is_einops_name(name) for name in inner):
                return None
            terms.append(tuple(inner))
            index = close + 1
            continue
        if char == ")":
            return None
        end = index
        while end < len(side) and not side[end].isspace() and side[end] not in "()":
            end += 1
        token = side[index:end]
        if token == "...":
            terms.append("...")
        elif _is_einops_name(token):
            terms.append(token)
        else:
            return None
        index = end
    return terms


def _is_einops_name(token: str) -> bool:
    return token == "1" or (token.isidentifier() and token != "_")


def _einops_shape(
    pattern: str,
    source: tuple[DimExpr, ...],
    sizes: dict[str, DimExpr],
) -> tuple[DimExpr, ...] | None:
    """Apply an einops ``rearrange`` pattern to a shape.

    The pattern says exactly what happens to each axis, so this needs no
    guessing: bind the left side's names to the source dims, then read the
    right side off those bindings. A group ``(a b)`` multiplies its axes
    together, and the ellipsis carries whatever axes the named ones did not
    claim. Returns *None* for anything it cannot account for exactly -- an
    unresolvable split, a name the right side introduces from nowhere -- so an
    unsupported pattern passes the tensor through rather than inventing a rank.
    """
    if "->" not in pattern:
        return None
    left_side, right_side = pattern.split("->", 1)
    left = _parse_einops_side(left_side)
    right = _parse_einops_side(right_side)
    if left is None or right is None:
        return None
    if left.count("...") > 1 or right.count("...") > 1:
        return None

    # The ellipsis absorbs every axis the named terms do not take.
    named = sum(1 for term in left if term != "...")
    if "..." in left:
        if len(source) < named:
            return None
        ellipsis_at = left.index("...")
        before = ellipsis_at
        after = named - before
        ellipsis_dims = source[before : len(source) - after]
    else:
        if len(source) != named:
            return None
        ellipsis_dims = ()

    bound: dict[str, DimExpr] = {}
    cursor = 0
    for term in left:
        if term == "...":
            cursor += len(ellipsis_dims)
            continue
        if cursor >= len(source):
            return None
        dim = source[cursor]
        cursor += 1
        if isinstance(term, tuple):
            # A grouped INPUT axis splits one dim into several. Every factor but
            # one must be given (``d=self.head_dim``); the remaining one is what
            # is left over.
            known = [name for name in term if name in sizes or name == "1"]
            unknown = [name for name in term if name not in known]
            if len(unknown) > 1:
                return None
            for name in known:
                bound[name] = 1 if name == "1" else sizes[name]
            if unknown:
                divisor = _merge_axes(tuple(bound[name] for name in known) or (1,))
                remainder = _divide_dim(dim, divisor)
                if remainder is None:
                    return None
                bound[unknown[0]] = remainder
        else:
            if term == "1":
                continue
            bound[term] = sizes.get(term, dim)

    out: list[DimExpr] = []
    for term in right:
        if term == "...":
            out.extend(ellipsis_dims)
        elif isinstance(term, tuple):
            factors = []
            for name in term:
                if name == "1":
                    factors.append(1)
                elif name in bound:
                    factors.append(bound[name])
                else:
                    return None
            merged = _merge_axes(tuple(factors))
            if merged is None:
                return None
            out.append(merged)
        elif term == "1":
            out.append(1)
        elif term in bound:
            out.append(bound[term])
        else:
            return None
    return tuple(out)


def _divide_dim(dim: DimExpr, divisor: DimExpr) -> DimExpr | None:
    """``dim / divisor`` as a shape dim, or None when it does not divide."""
    factors = list(_dim_factors(dim))
    for token in _dim_factors(divisor):
        if token in factors:
            factors.remove(token)
            continue
        numeric = [item for item in factors if item.lstrip("-").isdigit()]
        if not token.lstrip("-").isdigit() or not numeric:
            return None
        product = 1
        for item in numeric:
            product *= int(item)
        if product % int(token):
            return None
        product //= int(token)
        for item in numeric:
            factors.remove(item)
        if product != 1:
            factors.append(str(product))
    if not factors:
        return 1
    return _merge_axes(tuple(factors))


def _merge_axes(axes: tuple[DimExpr, ...]) -> DimExpr | None:
    """Multiply a contiguous span of shape axes into one flattened dim.

    Used by ``flatten`` (collapse ``[start_dim, end_dim]``). Pure-numeric spans
    return an ``int`` (``8``); spans with symbolic factors return a readable
    ``*``-joined product (``"K*P"``) with size-1 axes dropped. Tolerates both
    ``int`` and stringified numeric axes. ``None`` only for an empty span.
    """
    if not axes:
        return None
    num = 1
    syms: list[str] = []
    for dim in axes:
        for token in _dim_factors(dim):
            if token.lstrip("-").isdigit():
                num *= int(token)
            else:
                syms.append(token)
    if not syms:
        return num
    return "*".join(syms + ([str(num)] if num != 1 else []))


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


_STARRED_SHAPE_SLICE_RE = re.compile(
    r"^\*\s*[A-Za-z_][\w.]*\.shape\[(?P<slice>[^\]]*)\]$"
)


def _starred_shape_axes(
    token: str, template: TensorSpec | None
) -> list[DimExpr] | None:
    """Axes a ``*<tensor>.shape[...]`` size argument stands for.

    ``kv_nope.new_empty(*kv_nope.shape[:-1], head_dim)`` says "the same leading
    axes as this tensor, then a new last one". The starred term is not one size
    but however many the slice selects, so it has to be read off the template
    tensor rather than folded to a scalar. ``None`` when the token is not that
    form or no template is available.
    """
    match = _STARRED_SHAPE_SLICE_RE.match(token)
    if match is None or template is None or not template.shape:
        return None
    text = match.group("slice").strip()
    parts = text.split(":")
    if len(parts) == 1:
        index = _int_dim(parts[0])
        if index is None:
            return None
        try:
            return [template.shape[index]]
        except IndexError:
            return None
    if len(parts) > 3:
        return None
    bounds: list[int | None] = []
    for part in parts:
        part = part.strip()
        if not part:
            bounds.append(None)
            continue
        value = _int_dim(part)
        if value is None:
            return None
        bounds.append(value)
    while len(bounds) < 3:
        bounds.append(None)
    return list(template.shape[bounds[0] : bounds[1] : bounds[2]])


def _constructed_shape(
    details: Sequence[str],
    dims: dict[str, DimExpr],
    template: TensorSpec | None = None,
) -> tuple[DimExpr, ...] | None:
    """Axes of a tensor built from host-scalar sizes (``ones``/``zeros``/...).

    The extractor stamped one ``size<i>`` per positional size argument. Each
    resolves to an ``int`` literal, a known dim, or -- for a runtime length --
    its bare identifier kept as a symbolic extent, never ``?``. No sizes at all
    means the call was written in a form this does not read, so report nothing
    rather than invent a rank.

    *template* is the receiver of a ``new_*`` constructor, which such a call can
    name its leading axes from (``*x.shape[:-1]``).
    """
    if (
        _detail_value(details, "size0") is None
        and _detail_value(details, "sizes") == "()"
    ):
        # ``torch.full((), 0.0)`` builds a SCALAR. A 0-d ``()`` cannot be
        # reported -- the display formatter drops an empty shape and the node
        # would be re-filled with the section default -- so say ``[1]``, the
        # same stand-in the constant path uses.
        return (1,)
    axes: list[DimExpr] = []
    index = 0
    while True:
        token = _detail_value(details, f"size{index}")
        if token is None:
            break
        token = token.strip()
        starred = _starred_shape_axes(token, template)
        if starred is not None:
            axes.extend(starred)
            index += 1
            continue
        as_int = _int_dim(token)
        if as_int is not None:
            axes.append(as_int)
        else:
            resolved = _resolve_dim_name(token, dims)
            if resolved is None:
                # A size can be arithmetic over config scalars rather than one
                # name: GLM allocates its key buffer
                # ``self.qk_nope_head_dim + self.qk_rope_head_dim`` wide. Keeping
                # that as a symbolic string reports a head dim nothing can
                # compare, which is how a key of 256 read as 512.
                folded = _eval_dim_expr(token, dims)
                if folded is not None:
                    resolved = folded
            if resolved is not None:
                axes.append(resolved)
            else:
                bare = token
                for prefix in ("self.config.", "config.", "self."):
                    if bare.startswith(prefix):
                        bare = bare[len(prefix) :]
                        break
                axes.append(bare)
        index += 1
    return tuple(axes) if axes else None


def _with_explicit_dtype(node: ModelGraphNode, spec: TensorSpec) -> TensorSpec:
    """Honour a ``dtype=`` the op was explicitly given.

    An op handed a dtype produces it, whatever flowed in: GLM's
    ``seqlens.cumsum(dim=0, dtype=torch.int32)`` is what keeps ``cu_seqlens`` an
    int32 offset tensor. The rules that compute the shape reason about extents
    and pass the input's dtype through, so the declared one is applied here, in
    one place, rather than taught to each of them. Constructors already resolve
    their own (:func:`_constructed_dtype`) and agree with this.
    """
    declared = _constructed_dtype(node.metadata.get("details") or ())
    if declared is None or declared == spec.dtype:
        return spec
    return TensorSpec(shape=spec.shape, dtype=declared)


def _reduction_dtype(
    source_dtype: str, index_reduction: bool, boolean_reduction: bool
) -> str:
    """Dtype a reduction answers: an index is int64, a predicate is bool."""
    if index_reduction:
        return "int64"
    if boolean_reduction:
        return "bool"
    return source_dtype


def _reduction_axes(value: Any) -> tuple[int, ...] | None:
    """The axes a reduction's ``dim`` names, or ``None`` when it names none.

    A reduction may name one axis (``sum(dim=2)``) or several
    (``sum(dim=(-1, -2))``); ``dim=None`` names none and reduces everything.
    Returning only the first axis of a tuple leaves the others in the published
    shape.
    """
    if value is None:
        return None
    token = str(value).strip()
    if not token or token == "None":
        return None
    try:
        parsed = ast.literal_eval(token)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, int) and not isinstance(parsed, bool):
        return (parsed,)
    if isinstance(parsed, (tuple, list)) and parsed:
        axes = tuple(
            item
            for item in parsed
            if isinstance(item, int) and not isinstance(item, bool)
        )
        return axes if len(axes) == len(parsed) else None
    return None


def _as_int(value: Any) -> int | None:
    """``value`` as an int when it plainly is one, else ``None``."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _constructed_dtype(details: Sequence[str]) -> str | None:
    """The ``dtype=`` a constructor was given, as a bare torch dtype name."""
    token = _detail_value(details, "dtype")
    if not token:
        return None
    token = token.strip()
    # Only a LITERAL dtype says anything: ``dtype=torch.long`` names one, while
    # ``dtype=query_states.dtype`` just points at another tensor's, and reading
    # the last dotted segment of that would report a dtype called "dtype".
    if not token.startswith("torch."):
        return None
    name = token[len("torch.") :]
    return name or None


def _arange_axis(details: Sequence[str], dims: dict[str, DimExpr]) -> DimExpr:
    """Length of the 1-D axis produced by ``torch.arange(start, stop, step)``.

    Bounds are the expressions the extractor stamped (``arange_start`` /
    ``arange_stop`` / ``arange_step``). Each resolves to an ``int`` literal, a
    config dim, or -- for a runtime length like ``compressed_len``/``n_windows``
    -- its bare identifier kept as a symbolic extent (never ``?``). When every
    bound is an int the concrete length ``ceil((stop-start)/step)`` is folded;
    the common ``arange(N)`` (start 0, step 1) keeps ``N`` exactly.
    """

    def resolve(token: str | None, default: DimExpr) -> DimExpr:
        if token is None:
            return default
        token = token.strip()
        as_int = _int_dim(token)
        if as_int is not None:
            return as_int
        val = _resolve_dim_name(token, dims)
        if val is not None:
            return val
        bare = token
        for prefix in ("self.config.", "config.", "self."):
            if bare.startswith(prefix):
                bare = bare[len(prefix) :]
                break
        return bare

    start = resolve(_detail_value(details, "arange_start"), 0)
    stop = resolve(_detail_value(details, "arange_stop"), None)
    step = resolve(_detail_value(details, "arange_step"), 1)
    if (
        isinstance(start, int)
        and isinstance(stop, int)
        and isinstance(step, int)
        and step > 0
    ):
        return max(0, (stop - start + step - 1) // step)
    if stop is None:
        return 0
    # A symbolic bound: ``arange(N)`` is exactly ``N`` long; keep that extent.
    if start == 0 and step == 1:
        return stop
    return stop


def _shape_snapshot_tokens(details: Sequence[str]) -> frozenset[str]:
    """Reshape targets that came from a shape local bound earlier.

    The extractor marks these (``shape_snapshots:``) because such a token records
    what its tensor measured where the local was bound -- possibly a different
    tensor from the one being reshaped here.
    """
    token = _detail_value(details, "shape_snapshots")
    if not token:
        return frozenset()
    return frozenset(item.strip() for item in token.split(",") if item.strip())


def _resolve_view_shape(
    detail: str,
    source: TensorSpec,
    dims: dict[str, DimExpr],
    named_shapes: dict[str, tuple[DimExpr, ...]] | None = None,
    snapshots: frozenset[str] | None = None,
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
    # Whether the leading axes came from a tensor the caller named, rather than
    # from the tensor being reshaped. Only then is an unresolvable trailing axis
    # safe to settle by element conservation.
    used_named_prefix = False

    # Detect starred prefix like ``*foo.shape[:-1]`` or ``*foo.shape[:-N]``.
    first = parts[0]
    if first.startswith("*") and ".shape" in first:
        m = re.search(r"\.shape\[:\s*(-?\d+)\]", first)
        if m:
            cut = int(m.group(1))
            leading = source.shape[:cut] if cut < 0 else source.shape[:cut]
            # ``cut`` indexes the *referenced* tensor's rank (``X`` in
            # ``*X.shape[:cut]``), but ``source`` here can be higher-rank than
            # ``X``: the attention epilogue ``reshape(*hidden_states.shape[:-1],
            # -1)`` runs on the 4-D attn core ``[B, S, heads, head_dim]`` while
            # ``hidden_states`` is 3-D, so ``source.shape[:-1]`` keeps one axis
            # too many and the lone trailing ``-1`` would collapse a single dim
            # -- a no-op reshape the source never intends. When the star is
            # followed by exactly one ``-1`` (merge the whole trailing feature
            # block into one axis), drop extra leading dims so the ``-1`` spans a
            # real (>=2-dim) block. Unaffected when the naive cut already leaves a
            # multi-dim tail (rank(source) == rank(X)) or when explicit trailing
            # dims follow the star (the head-split view, which is not a no-op).
            if parts[1:] == ["-1"] and len(source.shape) > 2:
                while len(leading) >= 1 and len(source.shape[len(leading) :]) < 2:
                    leading = source.shape[: len(leading) - 1]
            # The star names a TENSOR, which is not always the one being
            # reshaped. When the caller knows that tensor's shape -- a frame's
            # own input, say -- those are the axes the model asked for, and the
            # source's are a different tensor's. GPT-2's `Conv1D` restores the
            # leading axes of the `[B, S, in]` it was handed from a `[B*S, nf]`
            # projection, so reading the source drops an axis.
            # Only for a SNAPSHOT token -- a shape local bound earlier, which
            # records what that tensor measured THEN. An inline
            # `*x.shape[:-1]` written in the reshape itself names the tensor
            # being reshaped, and reading elsewhere gave MiniMax's attention a
            # fourth axis and then a second batch axis.
            named_prefix = (
                (named_shapes or {}).get(first[1:].split(".shape", 1)[0].strip())
                if first[1:] in (snapshots or ())
                else None
            )
            if named_prefix is not None and -len(named_prefix) <= cut < 0:
                leading = tuple(named_prefix[:cut])
                used_named_prefix = True
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
        # ``x.shape[i]`` usually names the tensor BEING reshaped -- the leading
        # batch/seq axes a reshape preserves -- and then the source is the right
        # place to read it. GLM's vision tower does both in one module:
        #
        #   hidden_states.view(-1, m, m, hidden_states.shape[-1])  # itself
        #   attn_output.reshape(seq_length, -1)   # seq_length = hidden_states.shape[0]
        #
        # The second names a DIFFERENT tensor (the module's input, not the
        # attention output), and reading the source there took the attention's
        # leading 1 and collapsed the patch axis into the tail. So consult the
        # named tensor only for a SNAPSHOT token -- a shape local bound earlier
        # (``seq_length = hidden_states.shape[0]``), which records what that
        # tensor measured THEN. An inline ``hidden_states.shape[-1]`` written in
        # the reshape itself reads the local as it is now, which is the source.
        shape_ref = re.match(r"^(?P<name>[\w.]+)\.shape\[(?P<axis>-?\d+)\]$", part)
        if shape_ref is not None:
            axis = int(shape_ref.group("axis"))
            name = shape_ref.group("name")
            named = (
                (named_shapes or {}).get(name) if part in (snapshots or ()) else None
            )
            if named is not None and -len(named) <= axis < len(named):
                resolved.append(named[axis])
                continue
            if -len(source.shape) <= axis < len(source.shape):
                resolved.append(source.shape[axis])
                continue
        # A single unresolvable axis is still determined when the LEADING axes
        # came from a named tensor: those are known, and a reshape conserves
        # elements, which is exactly what ``-1`` means. GPT-2's `Conv1D`
        # restores `(*x.shape[:-1], self.nf)` where `nf` is a CONSTRUCTOR
        # argument no config states.
        #
        # Only then. Guessing an axis from a source whose leading dims are
        # themselves a fallback produces nonsense -- measured: GLM's
        # `[B, S, 4096]` collapsed to `[B*S*4096]` and MiniMax grew a second
        # batch axis. Returning None leaves those to a fallback that is right.
        if used_named_prefix and neg_index is None:
            neg_index = len(resolved)
            resolved.append("-1")
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


def _resolve_expand_shape(
    detail: str,
    source: TensorSpec,
    dims: dict[str, DimExpr],
) -> tuple[DimExpr, ...] | None:
    """Resolve ``Tensor.expand`` arguments into a concrete broadcast shape.

    ``expand`` broadcasts existing size-1 axes to a larger size; a ``-1`` (or a
    value matching the current axis) keeps that axis unchanged. This differs from
    ``view``/``reshape`` (whose ``-1`` means element conservation), so it needs
    its own resolver. Alignment is positional against ``source.shape``: on any
    rank mismatch — the args cannot be aligned axis-for-axis (e.g. the router's
    ``-1, 1, 288`` against a 4-D source) — return ``None`` so the caller passes
    the source through rather than corrupting its rank.
    """
    if not detail or not source.shape:
        return None
    parts = [p.strip() for p in detail.split(",") if p.strip()]
    if len(parts) != len(source.shape):
        return None  # cannot align axis-for-axis — pass through
    resolved: list[DimExpr] = []
    for part, current in zip(parts, source.shape):
        if part == "-1":
            resolved.append(current)
            continue
        try:
            target: DimExpr | None = int(part)
        except ValueError:
            target = _resolve_dim_name(part, dims)
        # Only a genuine size-1 axis broadcasts; every other axis keeps its
        # current (possibly symbolic) size — matching what a ``-1`` would do and
        # torch's requirement that a non-1 axis equal the requested size.
        if target is not None and current == 1 and target != 1:
            resolved.append(target)
        else:
            resolved.append(current)
    return tuple(resolved)


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
        Symbol.BATCH.value in inp.shape
        or any(isinstance(d, str) and "B" in str(d) for d in inp.shape)
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


# Shape behaviour of the Triton-language operations the kernel decomposer emits.
# Sanctioned as a shape-inference table: it records what each operation does to a
# shape, which is exactly the kind of knowledge shape inference is allowed to
# hold. It is NOT a list of recognised kernels or a display-name map -- nothing
# here decides whether something is a kernel or what it is called.
_TRITON_ELEMENTWISE_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "sigmoid",
        "sqrt",
        "exp",
        "exp2",
        "softplus",
    }
)
# Associative scans: same shape in, same shape out.
_TRITON_SCAN_OPS = frozenset({"cumsum", "cumprod"})


def _broadcast_shapes(shapes: list[tuple[DimExpr, ...]]) -> tuple[DimExpr, ...]:
    """Right-aligned NumPy-style broadcast of several shapes.

    A concrete size-1 axis yields to a larger sibling; otherwise the first
    non-``1`` axis (concrete or symbolic) is kept. Mismatched concrete non-1 axes
    are a modeling error we do not try to reconcile -- the first is kept. Used to
    size the index part of an integer advanced index ``base[i, j, ...]``.
    """
    if not shapes:
        return ()
    rank = max(len(shape) for shape in shapes)
    result: list[DimExpr] = []
    for offset in range(1, rank + 1):
        dim: DimExpr = 1
        for shape in shapes:
            if offset > len(shape):
                continue
            value = shape[-offset]
            if dim == 1:
                dim = value
        result.append(dim)
    return tuple(reversed(result))


def _sum_dim_sizes(sizes: list[DimExpr]) -> DimExpr:
    """Combine per-operand axis sizes (e.g. for ``concat``) into their sum.

    Sizes are ``int`` when concrete and ``str`` (e.g. ``"S*8"``) when symbolic.
    A single non-int size used to force the whole sum to bail out to "keep the
    widest operand's shape" (an identity concat), which is wrong whenever any
    contributor -- not just the widest one -- has a genuinely symbolic size.
    Fold every concrete size into one running total and join it with the
    symbolic terms, so ``["S*8", 1]`` -> ``"S*8 + 1"`` instead of silently
    dropping the ``1``.
    """
    if all(isinstance(size, int) for size in sizes):
        return sum(sizes)  # type: ignore[arg-type]
    total = 0
    parts: list[str] = []
    for size in sizes:
        if isinstance(size, int):
            total += size
        else:
            parts.append(str(size))
    if total:
        parts.append(str(total))
    parts = _cancel_subtracted_terms(parts)
    if not parts:
        return total if total else 0
    if len(parts) == 1 and str(parts[0]).isdigit():
        return int(parts[0])
    return " + ".join(parts) if parts else 0


def _cancel_subtracted_terms(parts: list[str]) -> list[str]:
    """Cancel a term against its own subtrahend: ``A-(B)`` summed with ``B`` is ``A``.

    ``rotate_half`` concatenates ``x[..., :d//2]`` with ``x[..., d//2:]``, whose
    widths are ``d//2`` and ``d-(d//2)``. Their sum is exactly ``d``, but with a
    symbolic ``d`` the two halves are strings, so the axis rendered as
    ``index_head_dim-(index_head_dim//2) + index_head_dim//2`` -- correct, and
    unreadable. This is ordinary cancellation, not simplification of arbitrary
    arithmetic: a part is dropped only when another part is *character for
    character* the thing it subtracts.
    """
    remaining = list(parts)
    for index, part in enumerate(remaining):
        match = re.fullmatch(r"(.+?)\s*-\s*\((.+)\)", str(part))
        if match is None:
            continue
        minuend, subtrahend = match.group(1), match.group(2)
        for other, candidate in enumerate(remaining):
            if other == index or str(candidate) != subtrahend:
                continue
            reduced = [
                value
                for position, value in enumerate(remaining)
                if position not in {index, other}
            ]
            return _cancel_subtracted_terms([minuend, *reduced])
    return remaining


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


def _parse_split_sizes(text: str, dims: dict[str, DimExpr]) -> list[DimExpr] | None:
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


def _multi_output_slice_shape(
    source: TensorSpec,
    details: list[str],
    operation_label: str,
    ordinal: int,
    dims: dict[str, DimExpr],
    output_count: int | None = None,
) -> TensorSpec | None:
    """Shape of one output slice of a split/chunk/unbind.

    ``split([4, 4, 16], dim=-1)`` gives ordinal 2 a ``[..., 16]`` slice; ``chunk``
    divides the dim evenly for every ordinal; ``unbind`` removes the split axis
    entirely (each output drops that dimension). Returns ``None`` when the source
    dim is symbolic/unknown so the caller can fall back.
    """
    if not source.shape:
        return source
    dim_str = _detail_value(details, "dim")
    default_dim = 0 if operation_label == "unbind" else -1
    dim = _int_dim(dim_str) if dim_str is not None else default_dim
    if dim is None:
        dim = default_dim
    resolved_dim = dim % len(source.shape)
    if operation_label == "unbind":
        # ``source`` may be the whole pre-unbind tensor (its unbind axis still
        # present, sized to the number of outputs) or -- when the trace only
        # recorded a single output tensor -- one already-materialised slice. Drop
        # the axis only in the former case; slicing an already-sliced spec a second
        # time would wrongly shed a real dimension ([Pv, 1024] -> [1024]).
        axis_size = source.shape[resolved_dim]
        whole = (
            output_count is not None
            and isinstance(axis_size, int)
            and axis_size == output_count
        )
        if not whole:
            return source
        new_shape = tuple(
            value for index, value in enumerate(source.shape) if index != resolved_dim
        )
        return TensorSpec(shape=new_shape, dtype=source.dtype)
    dim_val = source.shape[resolved_dim]
    split_size = _detail_value(details, "split_size")
    if not isinstance(dim_val, int) or split_size is None:
        return None
    if operation_label == "chunk":
        try:
            count = int(split_size)
        except (ValueError, TypeError):
            return None
        if count <= 0:
            return None
        return TensorSpec(
            shape=_replace_dim(source.shape, resolved_dim, dim_val // count),
            dtype=source.dtype,
        )
    sizes = _parse_split_sizes(split_size, dims)
    if not sizes:
        return None
    out_size = sizes[ordinal] if ordinal < len(sizes) else sizes[-1]
    return TensorSpec(
        shape=_replace_dim(source.shape, resolved_dim, out_size),
        dtype=source.dtype,
    )


def _model_declared_aliases(spec: ArchitectureSpec) -> dict[str, str]:
    """Renames the model's own config classes declare: read name -> real key.

    Modeling code reads ``config.num_local_experts`` against a checkpoint that
    spells it ``n_routed_experts``; the model says so itself in its config
    class's ``attribute_map``, which is read rather than guessed because a
    guessed list has to claim a name globally and ``n_heads`` is GLM's sparse
    indexer head count, not another word for its attention heads.
    """
    return declared_config_aliases(spec.code_paths or [])


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
                elif isinstance(
                    stmt, (_pyast.For, _pyast.While, _pyast.With, _pyast.Try)
                ):
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


def _function_ast_locals(
    func: ast.FunctionDef,
) -> dict[str, tuple[ast.AST, int | None]]:
    """Map local names to ``(value_expr, tuple_index)`` for a function's assignments.

    ``a = expr`` -> ``a: (expr, None)``; ``a, b = expr`` -> ``a: (expr, 0)``,
    ``b: (expr, 1)``. Used to chase a buffer tensor back through local aliases and
    tuple unpacks (e.g. ``inv_freq, self.scaling = rope_init_fn(...)``).
    """
    locals_map: dict[str, tuple[ast.AST, int | None]] = {}
    for stmt in ast.walk(func):
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None and isinstance(stmt.target, ast.Name):
                locals_map[stmt.target.id] = (stmt.value, None)
            continue
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                locals_map[target.id] = (stmt.value, None)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for index, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        locals_map[elt.id] = (stmt.value, index)
    return locals_map


def _resolved_scalar_locals(
    func: ast.FunctionDef, *, config: dict[str, Any], context: ShapeContext
) -> dict[str, DimExpr]:
    """Resolve a function's simple ``name = <dim expr>`` locals to concrete dims.

    Feeds ``torch.arange`` argument resolution (``spatial_dim = dim // 2``) when we
    hop into a RoPE init helper.
    """
    local_vars: dict[str, DimExpr] = {}
    for stmt in func.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    resolved = _resolve_dim_expr(
                        stmt.value,
                        config=config,
                        local_vars=local_vars,
                        context=context,
                    )
                    if resolved is not None:
                        local_vars[target.id] = resolved
    return local_vars


def _last_return_value(func: ast.FunctionDef) -> ast.AST | None:
    found: ast.AST | None = None
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            found = node.value
    return found


def _arange_length(
    call: ast.Call,
    *,
    config: dict[str, Any],
    local_vars: dict[str, DimExpr],
    context: ShapeContext,
) -> int | None:
    """Length of ``torch.arange(start, stop, step)`` when all bounds resolve to ints."""

    def resolve(node: ast.AST) -> DimExpr | None:
        return _resolve_dim_expr(
            node, config=config, local_vars=local_vars, context=context
        )

    positional = list(call.args)
    start: DimExpr | None = 0
    stop: DimExpr | None = None
    step: DimExpr | None = 1
    if len(positional) == 1:
        stop = resolve(positional[0])
    elif len(positional) >= 2:
        start = resolve(positional[0])
        stop = resolve(positional[1])
        if len(positional) >= 3:
            step = resolve(positional[2])
    for keyword in call.keywords:
        if keyword.arg == "start":
            start = resolve(keyword.value)
        elif keyword.arg == "end":
            stop = resolve(keyword.value)
        elif keyword.arg == "step":
            step = resolve(keyword.value)
    if not (isinstance(start, int) and isinstance(stop, int) and isinstance(step, int)):
        return None
    if step <= 0:
        return None
    return max(0, (stop - start + step - 1) // step)


def _resolved_self_dims(
    func: ast.FunctionDef,
    *,
    config: dict[str, Any],
    context: ShapeContext,
    dim_locals: dict[str, DimExpr],
) -> dict[str, DimExpr]:
    """``self.<attr> = <dim expr>`` in ``__init__``, resolved to concrete dims.

    A buffer is routinely sized by the attributes the constructor just set
    (``torch.zeros(config.vocab_size, self.top_k)``), which are not locals and so
    are invisible to the ordinary local resolver.
    """
    resolved: dict[str, DimExpr] = {}
    for stmt in func.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not (isinstance(target, ast.Attribute) and _is_self_attr(target)):
                continue
            dim = _resolve_dim_expr(
                stmt.value, config=config, local_vars=dim_locals, context=context
            )
            if dim is not None:
                resolved[target.attr] = dim
    return resolved


def _constructor_buffer_spec(
    node: ast.AST,
    *,
    config: dict[str, Any],
    context: ShapeContext,
    dim_locals: dict[str, DimExpr],
    self_dims: dict[str, DimExpr] | None = None,
) -> ModuleParameterSpec | None:
    """Shape and dtype of a buffer built by ``torch.zeros``/``ones``/``empty``/``full``.

    Sizes resolve the way every other extent does -- an int literal, a config
    value, or a local already resolved in ``__init__``. Returns ``None`` unless
    EVERY size resolves, so a buffer this cannot size is left alone rather than
    given a guess.
    """
    if not isinstance(node, ast.Call):
        return None
    name = _call_class_name(node)
    if name not in {"zeros", "ones", "empty", "full"}:
        return None
    positional = list(node.args)
    if name == "full":
        positional = positional[:1]
    sizes: list[ast.expr] = []
    for arg in positional:
        if isinstance(arg, (ast.Tuple, ast.List)):
            sizes.extend(arg.elts)
        else:
            sizes.append(arg)
    if not sizes:
        return None
    shape: list[DimExpr] = []
    for size in sizes:
        resolved: DimExpr | None = None
        if (
            self_dims
            and isinstance(size, ast.Attribute)
            and _is_self_attr(size)
            and size.attr in self_dims
        ):
            resolved = self_dims[size.attr]
        if resolved is None:
            resolved = _resolve_dim_expr(
                size, config=config, local_vars=dim_locals, context=context
            )
        if resolved is None:
            return None
        shape.append(resolved)
    dtype: str | None = None
    for keyword in node.keywords:
        if keyword.arg == "dtype":
            token = _literal_torch_dtype(keyword.value)
            if token is not None:
                dtype = token
    return ModuleParameterSpec(shape=tuple(shape), dtype=dtype)


def _literal_torch_dtype(node: ast.AST) -> str | None:
    """``torch.long`` -> ``int64``; anything else unnamed answers ``None``."""
    if not (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
    ):
        return None
    return {"long": "int64", "int": "int32", "bool": "bool"}.get(
        node.attr,
        node.attr if node.attr.startswith(("int", "float", "bfloat")) else None,
    )


def _resolve_buffer_length(
    node: ast.AST | None,
    *,
    class_node: ast.ClassDef,
    config: dict[str, Any],
    context: ShapeContext,
    ast_locals: dict[str, tuple[ast.AST, int | None]],
    dim_locals: dict[str, DimExpr],
    index: int | None = None,
    depth: int = 0,
) -> int | None:
    """Trace a 1-D buffer tensor back to a ``torch.arange`` and return its length.

    Follows local aliases and tuple unpacks (``ast_locals``), one same-class method
    hop (``rope_init_fn = self.compute_axial_rope_parameters``; the static
    ``compute_axial_rope_parameters`` return), and length-preserving elementwise
    wrappers (``.to(device)``, ``.float()``, ``1.0 / base ** (...)``). General for
    any RoPE-style buffer; returns ``None`` (skip) when the chain does not terminate
    in an ``arange``.
    """
    if node is None or depth > 8:
        return None
    if isinstance(node, ast.Name):
        bound = ast_locals.get(node.id)
        if bound is None:
            return None
        value, tuple_index = bound
        next_index = index if index is not None else tuple_index
        return _resolve_buffer_length(
            value,
            class_node=class_node,
            config=config,
            context=context,
            ast_locals=ast_locals,
            dim_locals=dim_locals,
            index=next_index,
            depth=depth + 1,
        )
    if isinstance(node, ast.Call):
        func = node.func
        fname = _call_class_name(node)
        if fname == "arange":
            return _arange_length(
                node, config=config, local_vars=dim_locals, context=context
            )
        # One same-class method hop: `self.method(...)` or an aliased method name.
        method_name: str | None = None
        if isinstance(func, ast.Attribute) and _is_self_attr(func):
            method_name = func.attr
        elif isinstance(func, ast.Name):
            alias = ast_locals.get(func.id)
            if (
                alias is not None
                and isinstance(alias[0], ast.Attribute)
                and _is_self_attr(alias[0])
            ):
                method_name = alias[0].attr
        if method_name is not None:
            method = _find_method(class_node, method_name)
            if method is None:
                return None
            returned = _last_return_value(method)
            if returned is None:
                return None
            target = returned
            selector = index
            if (
                isinstance(returned, ast.Tuple)
                and index is not None
                and index < len(returned.elts)
            ):
                target = returned.elts[index]
                selector = None
            return _resolve_buffer_length(
                target,
                class_node=class_node,
                config=config,
                context=context,
                ast_locals=_function_ast_locals(method),
                dim_locals=_resolved_scalar_locals(
                    method, config=config, context=context
                ),
                index=selector,
                depth=depth + 1,
            )
        if fname == "Buffer" and node.args:
            return _resolve_buffer_length(
                node.args[0],
                class_node=class_node,
                config=config,
                context=context,
                ast_locals=ast_locals,
                dim_locals=dim_locals,
                index=None,
                depth=depth + 1,
            )
        # Length-preserving tensor method, e.g. `inv_freq.to(device)` / `.float()`.
        if isinstance(func, ast.Attribute):
            return _resolve_buffer_length(
                func.value,
                class_node=class_node,
                config=config,
                context=context,
                ast_locals=ast_locals,
                dim_locals=dim_locals,
                index=index,
                depth=depth + 1,
            )
        return None
    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        for child in ast.iter_child_nodes(node):
            length = _resolve_buffer_length(
                child,
                class_node=class_node,
                config=config,
                context=context,
                ast_locals=ast_locals,
                dim_locals=dim_locals,
                index=None,
                depth=depth + 1,
            )
            if length is not None:
                return length
        return None
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
        return _eval_config_condition(test.left, config=config, local_vars=local_vars)
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
    # A local bound to a list literal, e.g. ``kernel_size = [t, p, p]`` then
    # ``nn.Conv3d(..., kernel_size=kernel_size)``. The init walker stores such
    # resolved lists as tuples in ``local_vars``.
    if isinstance(node, ast.Name):
        bound = local_vars.get(node.id)
        if isinstance(bound, tuple):
            return bound
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
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # ``getattr(config, 'head_dim', None) or hidden // heads`` — take the
        # first operand that resolves to a truthy int, matching ``or`` semantics.
        for value in node.values:
            resolved = _resolve_dim_expr(
                value, config=config, local_vars=local_vars, context=context
            )
            if isinstance(resolved, int) and resolved:
                return resolved
        return None
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


def _slice_bound_name(raw: str | None) -> tuple[str, bool] | None:
    """``(name, takes_head)`` for a trailing-axis slice bounded by a bare name.

    ``(..., :rotary_dim)`` takes the head of the axis and ``(..., rotary_dim:)``
    the tail. Returns *None* for anything else -- a numeric bound is folded by
    the branches above, and a bound we cannot even name is not narrowed at all.
    """
    if not raw:
        return None
    element = raw.strip().removeprefix("(").removesuffix(")").split(",")[-1].strip()
    if element.count(":") != 1:
        return None
    head, _, tail = element.partition(":")
    head, tail = head.strip(), tail.strip()
    if head and not tail and head.isidentifier():
        return head, False
    if tail and not head and tail.isidentifier():
        return tail, True
    return None


def _detail_value(details: Sequence[str], key: str) -> str | None:
    """Read a recorded call detail such as ``dim: -1``."""
    prefix = f"{key}:"
    for item in details:
        text = str(item).strip()
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return None


def _ast_const_int(node: ast.AST) -> int | None:
    """The non-negative int a constant subscript index carries (``freq[:, 1]``)."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ):
        return node.value if node.value >= 0 else None
    return None


_PLAIN_DIM_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OP_LINE_RE = re.compile(r"@op_l(\d+)_c")


def _op_line_of(node_id: str) -> int | None:
    """Source line encoded in an op node id (``...@op_l1766_c32_unsqueeze...``)."""
    match = _OP_LINE_RE.search(node_id)
    return int(match.group(1)) if match else None


def _int_dim(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


_SHAPE_EXPR_OP_SYMBOLS: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "//",
    ast.Div: "//",
}


def _eval_shape_expr(expr: str, shape: Sequence[Any]) -> int | str | None:
    """Evaluate a slice-bound expression over a ``shape`` symbol.

    ``expr`` is emitted by ``ast_analyze._shape_relative_expr`` and references only a
    ``shape`` list, int constants, and ``+ - * // /`` arithmetic (e.g.
    ``shape[-1] // 2``).

    When every referenced dim is a concrete int, this folds to an exact int (the
    original behavior). When a referenced dim is symbolic (a non-numeric name
    like ``'S'``), the arithmetic can't be folded to a number, but it must still
    produce a result DISTINCT from the plain dim name so a narrowing slice over a
    symbolic axis doesn't look like a no-op to callers -- so this renders a
    symbolic expression string instead (e.g. ``shape[-1] // 2`` over a symbolic
    ``'S'`` dim yields ``'S//2'``).

    Returns ``None`` when the expression is empty or contains anything outside
    that safe grammar — the caller then leaves the axis unchanged.
    """
    if not expr:
        return None

    def _eval(node: ast.AST) -> int | str | None:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = _eval(node.operand)
            if operand is None:
                return None
            if isinstance(node.op, ast.UAdd):
                return operand
            return -operand if isinstance(operand, int) else f"-({operand})"
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            if left is None or right is None:
                return None
            op_symbol = _SHAPE_EXPR_OP_SYMBOLS.get(type(node.op))
            if op_symbol is None:
                return None
            if isinstance(left, int) and isinstance(right, int):
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if right == 0:
                    return None
                return left // right
            # At least one operand is a symbolic dim name: render a distinct
            # symbolic expression instead of failing (which would silently
            # leave the axis unchanged and hide a real narrowing).
            return f"{left}{op_symbol}{right}"
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "shape"
        ):
            idx = _eval(node.slice)
            if not isinstance(idx, int) or not shape:
                return None
            try:
                raw = shape[idx]
            except IndexError:
                return None
            numeric = _int_dim(raw)
            return numeric if numeric is not None else str(raw)
        return None

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    return _eval(tree)


def _descriptor_frame_specs(spec: "ArchitectureSpec") -> dict[str, TensorSpec]:
    """Frames that read a vision grid descriptor, to the spec of that descriptor.

    A vision-tower forward parameter the caller hands straight from its own
    signature (``self.visual(pixel_values, grid_thw=image_grid_thw)``) is a model
    input; see :func:`vision_tower_passthrough_inputs`. The frames it is passed to
    carry it as ``input_label``, and their ops read it with no predecessor in the
    graph -- so the fallback hands them the tower activation and the whole chain
    computes patch geometry instead of grid geometry.

    The descriptor's width is measured from the columns its readers take
    (``grid[:, 0]``, ``[:, 1]``, ``[:, 2]`` is three wide); nothing else in the
    model states it. Rows are :attr:`Symbol.VISION_GRID`, one per image.
    """
    passthrough = vision_tower_passthrough_inputs(spec)
    if not passthrough:
        return {}
    # Frame attr name -> the parameter it was handed, and per PARAMETER the widest
    # column anyone selects. The width belongs to the tensor, not to the frame:
    # one helper may read only ``grid[:, 0]`` while its sibling reads all three,
    # and the grid is three wide in both.
    owner: dict[str, str] = {}
    widest: dict[str, int] = {}
    for _attr, tree in spec.export_block_trees:
        pending = [tree]
        while pending:
            node = pending.pop()
            children = list(getattr(node, "children", None) or [])
            pending.extend(children)
            label = getattr(node, "input_label", None)
            attr_name = str(getattr(node, "attr_name", "") or "")
            if label not in passthrough or not attr_name:
                continue
            owner[attr_name] = label
            nested = list(children)
            while nested:
                inner = nested.pop()
                nested.extend(list(getattr(inner, "children", None) or []))
                for detail in getattr(inner, "details", None) or []:
                    if not str(detail).startswith("select_index:"):
                        continue
                    for token in str(detail).split(":", 1)[1].split(","):
                        try:
                            index = int(token.strip())
                        except ValueError:
                            continue
                        if index >= 0:
                            widest[label] = max(widest.get(label, 0), index + 1)
    return {
        attr_name: TensorSpec(
            shape=(Symbol.VISION_GRID.value, widest.get(label, 1)),
            dtype=passthrough[label] or "int64",
        )
        for attr_name, label in owner.items()
    }


def _vision_patch_flat_dim(vision_config: dict[str, Any]) -> int | None:
    """Flattened per-patch feature length fed to a vision tower's patch embed.

    A ``Glm5NextVisionPatchEmbed`` receives ``[num_patches, in_channels *
    temporal_patch_size * patch_size**2]`` and immediately ``view``s it back to
    ``[num_patches, in_channels, temporal_patch_size, patch_size, patch_size]``.
    Returns ``None`` (caller falls back to the generic ``(B, S, H)`` input) when any
    factor is missing, so non-GLM towers are never corrupted.
    """
    in_channels = _int_dim(vision_config.get("in_channels"))
    temporal = _int_dim(vision_config.get("temporal_patch_size"))
    patch = _int_dim(vision_config.get("patch_size"))
    if in_channels is None or temporal is None or patch is None:
        return None
    return in_channels * temporal * patch * patch


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
) -> (
    ModuleLinearSpec | ModuleConvSpec | ModuleEmbeddingSpec | ModuleParameterSpec | None
):
    if isinstance(node, ast.IfExp):
        # Conditional module assignment, e.g.
        # ``self.q_a_proj = nn.Linear(...) if q_lora_rank is not None else nn.Identity()``.
        # Resolve to whichever branch is a recognizable module constructor so the
        # real submodule (not the Identity fallback) supplies in/out dims.
        for branch in (node.body, node.orelse):
            spec = _parse_module_ctor(
                branch, config=config, local_vars=local_vars, context=context
            )
            if spec is not None:
                return spec
        return None
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
    """Last-resort width for a Linear whose own constructor could not be found.

    :meth:`_lookup_linear_spec` reads the real ``nn.Linear(in, out)`` from the
    owning class, which is where this should come from. It misses only where the
    owning class cannot be resolved -- a projection inside an expert list -- and
    then these names are all that is left.

    Measured against the pinned models, exactly these answer for something:
    ``gate_proj``, ``up_proj``, ``down_proj``, ``gate_up_proj`` and the routed
    expert projections. The rest of what this used to list -- ``lm_head``,
    ``w1``/``w2``/``w3``, ``router``, and ``q_proj``/``k_proj``/``v_proj``/
    ``o_proj`` -- answered for nothing, and the attention ones were wrong as
    well as unused: under grouped-query or latent attention a ``q_proj`` is
    ``heads * head_dim`` and a ``k_proj`` narrower still, neither of them the
    hidden width this claimed.
    """
    if not attr:
        return None
    lowered = attr.lower()
    if lowered in {"gate_proj", "up_proj"}:
        return context.dims.get(Symbol.INTERMEDIATE.value)
    if lowered == "down_proj":
        return context.dims.get(Symbol.HIDDEN.value)
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


def _slices_tile(
    whole: TensorSpec, slices: list[TensorSpec], *, removes_axis: bool
) -> bool:
    """True when *slices* genuinely divide *whole* along exactly one axis.

    A split's recorded sizes are only trustworthy if they account for the parent:
    the slices must agree with it on every axis but one, and their extents along
    that axis must add back up to it. When a size could not be resolved to a
    concrete integer the sum cannot be checked, so the shape is accepted only if
    it is otherwise consistent -- a symbolic width is normal, a width that
    contradicts the parent is not.

    ``unbind`` removes its axis instead of dividing it, so each slice has one
    fewer axis and the count of slices is what must match the parent's extent.
    """
    if not slices or not whole.shape:
        return False
    if removes_axis:
        return all(len(item.shape) == len(whole.shape) - 1 for item in slices)
    if any(len(item.shape) != len(whole.shape) for item in slices):
        return False
    differing = [
        axis
        for axis in range(len(whole.shape))
        if any(item.shape[axis] != whole.shape[axis] for item in slices)
    ]
    if len(differing) > 1:
        return False
    if not differing:
        # Every slice equals the parent: nothing was actually divided, so the
        # slices carry no information the parent does not already have.
        return False
    axis = differing[0]
    extents = [item.shape[axis] for item in slices]
    if any(isinstance(size, int) and size <= 0 for size in extents):
        return False
    if not all(isinstance(size, int) for size in extents):
        return True
    parent = whole.shape[axis]
    if not isinstance(parent, int):
        return True
    return sum(extents) == parent
