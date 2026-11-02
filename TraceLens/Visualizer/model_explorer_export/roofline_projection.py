###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Static, trace-free full-model roofline projection.

Walks the same merged Model Explorer graph the visualizer draws (see
:func:`TraceLens.Visualizer.model_explorer_export.ast_build.build_model_explorer_payload`),
stopped at ops with a direct :mod:`TraceLens.PerfModel.perf_model` formula (see
:func:`TraceLens.ModelUtils.build_roofline_basic_ops`), and costs each leaf with
PerfModel's analytical FLOPs/bytes. Reusing the graph rather than a separate
operator export means a ``--tp``/``--pp``/``--ep`` split, and the merged
per-GPU local shapes it produces, is the same split and the same shapes the
visualizer draws -- not a second, independently-matched approximation of it.

Op coverage is intentionally narrow: GEMM (``Linear``/``matmul``/``bmm``), CONV,
SDPA, the Normalization family, Softmax, generic elementwise/reduction ops, and
MoE expert routing -- exactly the leaf granularity
:func:`TraceLens.ModelUtils.build_roofline_basic_ops` stops AST expansion at,
plus the vLLM/SGLang engine boundaries (``ParallelLMHead``, ``FusedMoEFactory``)
that never expand at all (see "Known approximations" below). Convolutions in a
graph node are costed only when the meta-device pass threaded the kernel
geometry onto the node's ``details`` attr (``conv_weight=[...]``); the
input/output shapes alone cannot recover the weight's kernel size or groups.
A depthwise short conv packed inside a fused attention leaf is the exception:
its kernel comes from the checkpoint (``sconv_kernel_size`` /
``conv_kernel_size``) because that leaf has no conv node of its own. Anything
else (embeddings, a conv with no threaded geometry and no declared short-conv
kernel) is reported as "unmodeled" rather than silently dropped, so the
projection's coverage is always visible instead of assumed.

Convolutional models are handled separately. The graph expands a non-uniform
stage once (one conv node per stage, plus the stem and each stage's leading
downsampling conv). :func:`conv_ops_from_meta` still costs every live conv
instance a meta-device forward captures (see
:func:`TraceLens.ModelUtils.meta_trace.trace_meta_conv_ops`) -- each already
carries the concrete ``[B, C, H, W]`` shapes and weight geometry CONV needs --
and the CLI lets those supersede the graph's conv rows so a repeated stage is
not counted twice.

Known approximations, stated once here rather than scattered in comments:
  - GEMM bias is assumed absent (bias FLOPs are O(M*N), negligible next to the
    O(M*N*K) matmul term for any realistic K).
  - Every Normalization op is assumed affine; RMSNorm is assumed bias-free,
    every other norm is assumed to have a bias.
  - SDPA is assumed causal exactly when the query and key/value sequence
    lengths resolve equal (self-attention prefill shape), non-causal otherwise.
  - An op's instance count is the innermost ``"<N>x_..."`` repeat-count
    namespace segment surrounding it (the merged graph's own per-variant tile
    count), or 1 for an op outside any repeated layer. A hybrid/variant layer
    mix (sliding-window vs. global attention, MoE inserted only every few
    layers, ...) is therefore attributed per-variant already, unlike a flat
    ``num_hidden_layers`` scale-up.
  - ``ParallelLMHead``/``FusedMoEFactory`` (vLLM/SGLang) are read from the
    engine's own source, not the model's, and shape_inference has no rule for
    either -- their exported shape is an unresolved pass-through, not a real
    one. Both are costed from the architecture's own dims (batch, seq, vocab,
    hidden, experts-per-token, expert intermediate size) instead, the same
    numbers the model's own config declares, rather than from their (unusable)
    node shape.
  - MoE routing is costed as ``experts_per_tok`` grouped-GEMM passes per token
    (the gate_up_proj and down_proj of only the SELECTED experts), never
    ``num_experts`` passes -- the router picks a subset per token, so costing
    every expert for every token would overstate FLOPs by
    ``num_experts / experts_per_tok``. The expert GEMMs use :class:`GroupedGemm`
    over ``G = min(num_experts, active_tokens)`` distinct weight matrices, so the
    weight-byte term reflects the experts actually streamed (``G`` falls back to
    1, the prior single-weight cost, when ``num_experts`` is unknown). Tensor
    parallel column-shards each expert's intermediate width; expert parallel
    gives one GPU ``1/EP`` of the routed rows and adds the dispatch/combine
    all-to-all as FLOPs-free data movement (:class:`MoEComm`). The router's own
    softmax/top-k selection and the shared-expert add are not separately costed
    (both are O(tokens), negligible next to the expert GEMMs). A dense MLP that
    shares the same attribute (the layers below a ``dense_mlp_idx`` threshold)
    is costed as a SwiGLU over every token at the dense intermediate width, not
    as that routed subset.
  - A depthwise short convolution packed inside a fused attention leaf is
    costed when the config declares its kernel (``sconv_kernel_size`` /
    ``conv_kernel_size``): ``2 * tokens * channels * kernel``. A graph conv
    node uses the threaded ``conv_weight`` geometry instead.
  - ``num_predicted_tokens`` adds a secondary MTP / speculative draft stack:
    one draft depth per predicted token (capped at the depth the config
    declares), each a full attention + dense MLP + the 2H→H input projection,
    not a copy of the routed backbone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from math import prod
from typing import Any

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.loader import build_roofline_basic_ops
from TraceLens.ModelUtils.shape_inference import ShapeContext, Symbol
from TraceLens.PerfModel.perf_model import (
    CONV,
    GEMM,
    SDPA,
    BinaryElementwise,
    GroupedGemm,
    Normalization,
    Reduce,
    Softmax,
    UnaryElementwise,
)
from TraceLens.PerfModel.utils import torch_dtype_map
from TraceLens.TreePerf.tree_perf import compute_roofline_time_and_bound
from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.shapes import parse_shape_dims

# Constants

# Bytes per element by TraceLens.PerfModel.utils.torch_dtype_map's normalized
# simulation-precision name. torch_dtype_map already maps every spelling of a
# dtype TraceLens' dtype strings use (float16/bfloat16/fp8_e4m3/...) onto one of
# these; name2bpe duplicates this table but keyed on trace-event (c10::) names,
# which operator-export dtype strings are not.
_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp64": 8,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp4": 0.5,
}

# GEMM computation labels, split by how the op reaches the export: an actual
# nn.Module (operation == "nn_module") vs. a bare functional call.
_GEMM_NN_MODULE_COMPUTATIONS = frozenset({"Linear"})
_GEMM_FUNCTIONAL_LABELS = frozenset({"matmul", "batchmatmul", "mm", "bmm"})

# Convolution module computations. The input/output shapes are on the node; the
# weight shape (kernel geometry + groups) and the bias/transposed flags are not
# derivable from them, so they are threaded on the node's ``details`` attr by the
# meta-device pass (``conv_weight=[...]``, ``conv_bias=0|1``, ``conv_transposed=0|1``)
# and parsed by ``_parse_conv_details``.
_CONV_COMPUTATIONS = frozenset(
    {
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    }
)

# A vLLM/SGLang ``--engine`` draw labels a sharded linear by its own class
# (``RowParallelLinear``) or, for a model-specific subclass TraceLens'
# BasicOpFilter.for_roofline() still stops at
# (``MinimaxM3QKVParallelLinearWithIndexer``), by its own attr_name instead --
# never the plain ``"Linear"`` op_type a transformers-native draw uses. Any of
# these class names still names a weight-times-activation GEMM; its per-GPU
# local shape is already reflected on this node the same way a transformers
# ``nn.Linear`` leaf's is (see parallel_plan._localize_gpu_shapes), so the same
# GEMM formula applies once the label is recognized as one -- except
# ``ParallelLMHead``, read from the engine's own source where shape_inference
# has no rule for it; its M/N/K come from the architecture's dims instead (see
# _engine_head_cost).
_ENGINE_LINEAR_CLASS_MARKERS = ("ParallelLinear", "ParallelLMHead", "ReplicatedLinear")
_ENGINE_HEAD_CLASS_MARKERS = ("ParallelLMHead",)
# vLLM's ``FusedMoEFactory`` (SGLang's own MoE factory is a differently-named
# function call TraceLens does not expand either) builds a whole routing +
# expert-GEMM + combine pipeline across several more files; expanding it would
# trade today's one labeled boundary for a deep, mostly quantization/EPLB-
# plumbing subgraph with no clearer a picture of the actual compute. It stays
# a boundary, costed analytically instead (see _moe_cost).
_ENGINE_MOE_CLASS_MARKERS = ("FusedMoE",)

# Normalization family, keyed by a case-insensitive substring of the exported
# computation label: (kind, has_bias, channel_first). Matched by substring
# rather than exact class name because real checkpoints almost always subclass
# norm layers (LlamaRMSNorm, Qwen2RMSNorm, CustomRMSNorm, ...). "rmsnorm" has
# no learned bias; every other family here is assumed to have one. "generic"
# families share the base Normalization.flops_func/bytes_func; "rmsnorm" uses
# RMSNorm's own flops()/bytes() (see _rmsnorm_flops/_rmsnorm_bytes -- RMSNorm
# has no flops_func/bytes_func to call directly). channel_first classes
# normalize over axis 1 (conv-style [N, C, ...]); the rest normalize over the
# trailing feature axis.
_NORMALIZATION_FAMILIES: tuple[tuple[str, str, bool, bool], ...] = (
    ("rmsnorm", "rmsnorm", False, False),
    ("layernorm", "generic", True, False),
    ("groupnorm", "generic", True, True),
    ("instancenorm", "generic", True, True),
    ("batchnorm", "generic", True, True),
)


def _classify_normalization(computation: str) -> tuple[str, bool, bool] | None:
    """Map a (possibly model-specific subclass) norm label to its formula family."""
    lowered = computation.lower()
    for needle, kind, has_bias, channel_first in _NORMALIZATION_FAMILIES:
        if needle in lowered:
            return kind, has_bias, channel_first
    return None


_SDPA_COMPUTATION = "scaled_dot_product_attention"
_SOFTMAX_COMPUTATION = "Softmax"

# Pointwise/reduction computation labels mapped to UnaryElementwise/
# BinaryElementwise/Reduce. Shape-shuffling ops (reshape/transpose/concat/
# index/...) are deliberately left out of these sets -- they are usually free
# views or fused into a neighboring kernel, and PerfModel has no formula for
# them; they fall into "unmodeled" rather than being charged a cost they may
# not actually incur.
_UNARY_POINTWISE_COMPUTATIONS = frozenset(
    {
        "sigmoid",
        "silu",
        "gelu",
        "relu",
        "tanh",
        "softplus",
        "exp",
        "log",
        "log1p",
        "sqrt",
        "rsqrt",
        "square",
        "abs",
        "negate",
        "neg",
        "reciprocal",
        "sign",
        "cosine",
        "sine",
        "cos",
        "sin",
        "erf",
        "clamp",
        "cast",
    }
)
_BINARY_POINTWISE_COMPUTATIONS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "power",
        "maximum",
        "minimum",
        "elementwise_add",
        "elementwise_mul",
    }
)
_REDUCTION_COMPUTATIONS = frozenset(
    {
        "sum",
        "mean",
        "max",
        "min",
        "argmax",
        "argmin",
        "logsumexp",
        "norm",
        "variance",
        "std",
        "any",
        "all",
        "product",
    }
)

# A node's instance count is the innermost repeat-count namespace segment
# around it. A uniform stack is ``45x_DecoderLayer``. A non-uniform stage is
# ``s2_27x_ConvNextLayer``: the stage index keeps the containers distinct, and
# the count is the digits before ``x_``.
_REPEAT_NAMESPACE_SEGMENT = re.compile(r"^(?:s\d+_)?(\d+)x(?:_|$)")

# Ids this synthetic/boundary-marker prefix never carries real computation --
# an @input/@output port marker or a collective TraceLens inserted for a
# --tp/--pp/--ep split (modeling its own roofline cost is out of scope here;
# TreePerf's trace-driven comm model already covers collectives separately).
_NON_COMPUTE_ID_MARKER = "@comm:"


class DimResolutionError(ValueError):
    """A symbolic shape dimension has no concrete value."""


@dataclass
class RooflineOp:
    """One exported op's roofline projection."""

    name: str
    section: str
    computation: str
    operation: str
    category: str | None
    """PerfModel family name (``"GEMM"``, ``"SDPA"``, ...), or ``None`` when unmodeled."""
    modeled: bool
    reason: str | None
    """Why this op is unmodeled (``None`` when ``modeled`` is ``True``)."""
    output_elements: int | None
    instances: int
    """How many real layers this op's section represents (layer-repeat scaling)."""
    flops: float | None = None
    bytes: float | None = None
    compute_spec: str | None = None
    roofline_time_us: float | None = None
    roofline_bound: str | None = None

    @property
    def total_flops(self) -> float | None:
        return None if self.flops is None else self.flops * self.instances

    @property
    def total_bytes(self) -> float | None:
        return None if self.bytes is None else self.bytes * self.instances

    @property
    def total_roofline_time_us(self) -> float | None:
        return (
            None
            if self.roofline_time_us is None
            else self.roofline_time_us * self.instances
        )


@dataclass
class RooflineProjection:
    """Full-model roofline projection: every op plus a model-level rollup."""

    ops: list[RooflineOp] = field(default_factory=list)
    batch_size: int = 1
    seq_len: int = 0

    @property
    def modeled_ops(self) -> list[RooflineOp]:
        return [op for op in self.ops if op.modeled]

    @property
    def unmodeled_ops(self) -> list[RooflineOp]:
        return [op for op in self.ops if not op.modeled]

    def totals(self) -> dict[str, Any]:
        """Model-level rollup: FLOPs, bytes, roofline time, and coverage."""
        modeled = self.modeled_ops
        total_flops = sum(op.total_flops or 0 for op in modeled)
        total_bytes = sum(op.total_bytes or 0 for op in modeled)
        bound_ops = [op for op in modeled if op.roofline_time_us is not None]
        total_roofline_time_us = sum(op.total_roofline_time_us or 0 for op in bound_ops)
        compute_bound_time_us = sum(
            op.total_roofline_time_us or 0
            for op in bound_ops
            if op.roofline_bound == "COMPUTE_BOUND"
        )
        memory_bound_time_us = sum(
            op.total_roofline_time_us or 0
            for op in bound_ops
            if op.roofline_bound == "MEMORY_BOUND"
        )
        modeled_elements = sum(
            (op.output_elements or 0) * op.instances for op in modeled
        )
        unmodeled_elements = sum(
            (op.output_elements or 0) * op.instances for op in self.unmodeled_ops
        )
        total_elements = modeled_elements + unmodeled_elements
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "total_roofline_time_us": total_roofline_time_us,
            "compute_bound_time_us": compute_bound_time_us,
            "memory_bound_time_us": memory_bound_time_us,
            "modeled_op_count": len(modeled),
            "unmodeled_op_count": len(self.unmodeled_ops),
            "modeled_output_element_share": (
                modeled_elements / total_elements if total_elements else None
            ),
        }


def resolve_dim(dim: int | str, dims: dict[str, Any]) -> int:
    """Resolve one shape dimension (an int, a symbol, or a ``*``/``+`` expression).

    A negative int (an unfolded ``-1`` reshape placeholder) is treated as
    unresolved rather than used literally, since a literal ``-1`` would
    silently corrupt any FLOPs/bytes product it enters.
    """
    if isinstance(dim, int):
        if dim < 0:
            raise DimResolutionError(str(dim))
        return dim
    text = str(dim)
    if "+" in text:
        return sum(resolve_dim(term, dims) for term in text.split("+"))
    if "*" in text:
        result = 1
        for factor in text.split("*"):
            result *= resolve_dim(factor, dims)
        return result
    if text.lstrip("-").isdigit():
        return int(text)
    if text not in dims:
        raise DimResolutionError(text)
    value = dims[text]
    if value == text:
        raise DimResolutionError(text)
    return resolve_dim(value, dims)


def resolve_shape(shape: list[Any], dims: dict[str, Any]) -> tuple[int, ...]:
    return tuple(resolve_dim(dim, dims) for dim in shape)


def build_projection_dims(
    spec: ArchitectureSpec,
    *,
    batch_size: int,
    seq_len: int,
    extra_dims: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Concrete dims dict: config dims plus batch/seq and any ``--dim`` overrides."""
    context = ShapeContext.from_spec(spec)
    dims: dict[str, Any] = dict(context.dims)
    dims["B"] = batch_size
    dims["S"] = seq_len
    dims.update(extra_dims or {})
    return dims


def project_roofline(
    spec: ArchitectureSpec,
    *,
    batch_size: int,
    seq_len: int,
    extra_dims: dict[str, int] | None = None,
    gpu_arch: dict[str, Any] | None = None,
    num_predicted_tokens: int = 0,
    checkpoint: str | Path | None = None,
) -> RooflineProjection:
    """Project a kernel-by-kernel roofline for *spec* at a concrete batch/seq length.

    Builds the same merged Model Explorer graph the visualizer draws, with
    expansion stopped at roofline-formula ops (:func:`build_roofline_basic_ops`)
    and, when ``spec.parallel_layout`` is set, the same ``--tp``/``--pp``/``--ep``
    split and per-GPU local shapes the visualizer would show for that layout.

    ``num_predicted_tokens`` appends a secondary MTP / speculative draft stack,
    one depth per token, capped at the depth the checkpoint declares. ``0``
    (the default) costs only the backbone.
    """
    dims = build_projection_dims(
        spec, batch_size=batch_size, seq_len=seq_len, extra_dims=extra_dims
    )
    _populate_introspection(spec)
    moe_top_k = spec.num_experts_per_tok
    moe_intermediate_size = spec.moe_intermediate_size or spec.intermediate_size
    payload = build_model_explorer_payload(
        spec,
        basic_ops=build_roofline_basic_ops(),
        include_shapes=True,
        # The nested stage tree and the concrete spatial shapes are read off a
        # meta-device module. Without the checkpoint the projection builds the
        # uniform-stack graph and a CNN's stage body never appears.
        meta_shapes_checkpoint=str(checkpoint) if checkpoint else None,
    )
    graphs = payload["graphCollections"][0]["graphs"]
    nodes = graphs[0]["nodes"] if graphs else []

    result = RooflineProjection(batch_size=batch_size, seq_len=seq_len)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        parsed = _parse_op_node(node)
        if parsed is None:
            continue
        (
            section,
            computation,
            class_name,
            operation,
            output_text,
            input_texts,
            role,
            details,
        ) = parsed
        instances = _repeat_count(str(node.get("namespace") or ""), node)
        node_id = str(node.get("id"))
        op = _project_op(
            node_id,
            section,
            computation,
            class_name,
            operation,
            output_text,
            input_texts,
            dims,
            instances,
            gpu_arch,
            moe_top_k,
            moe_intermediate_size,
            role,
            details,
        )
        result.ops.append(op)
        # A fused attention leaf is costed by _attention_boundary_cost at only its
        # SDPA core; its packed Q/K/V/O(/R) projections live inside the opaque
        # block as no separate Linear nodes. Add them (and any relative-attention
        # term) as their own rows so the projection GEMMs are not silently dropped.
        if role == "attention" and op.modeled and op.category == "Attention":
            _dims_only, dtype = _split_shape_text(output_text)
            bpe = _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
            result.ops.extend(
                _attention_extra_ops(
                    section, node_id, instances, dims, bpe, dtype, gpu_arch
                )
            )
    if num_predicted_tokens > 0:
        result.ops.extend(
            _mtp_stack_ops(
                dims,
                num_predicted_tokens,
                gpu_arch,
                _activation_bpe(spec),
            )
        )
    return result


def _repeat_count(namespace: str, node: dict[str, Any] | None = None) -> int:
    """How many times the exported node stands for a live module.

    A ``repeat_count`` attr wins. The stage body stamps its depth there, and a
    submodule that exists only on the first element (a ResNet projection
    shortcut) stamps ``1`` so the stage depth does not multiply it. Otherwise
    the innermost ``{N}x_`` / ``s{stage}_{N}x_`` namespace segment, or 1.
    """
    stamped = _attr_text((node or {}).get("attrs"), "repeat_count")
    if stamped and str(stamped).isdigit():
        return int(stamped)
    count = 1
    for segment in namespace.split("/"):
        match = _REPEAT_NAMESPACE_SEGMENT.match(segment)
        if match:
            count = int(match.group(1))
    return count


def _attr_text(attrs: Any, key: str) -> str | None:
    if not isinstance(attrs, list):
        return None
    return next(
        (item.get("value") for item in attrs if item.get("key") == key),
        None,
    )


def _parse_op_node(
    node: dict[str, Any],
) -> tuple[str, str, str, str, str, list[list[str]]] | None:
    """``(section, computation, class_name, operation, output_shape_text,
    input_shape_lists)`` for *node*.

    ``None`` for a node that carries no computation of its own: an
    ``@input``/``@output`` boundary port, a host-side scalar source, or a
    collective TraceLens inserted for a device-partition split.
    """
    node_id = str(node.get("id") or "")
    if _NON_COMPUTE_ID_MARKER in node_id:
        return None
    attrs = node.get("attrs")
    computation = _attr_text(attrs, "op_type")
    if not computation:
        return None
    class_name = _attr_text(attrs, "class_name") or ""
    if any(marker in class_name for marker in _ENGINE_LINEAR_CLASS_MARKERS):
        computation = "Linear"
    operation = _attr_text(attrs, "operation") or ""
    output_text = _attr_text(attrs, "output_shape")
    if not output_text:
        return None
    input_shapes_json = _attr_text(attrs, "input_shapes")
    input_texts: list[list[str]] = (
        json.loads(input_shapes_json) if input_shapes_json else []
    )
    namespace = str(node.get("namespace") or "")
    section = namespace.split("/")[-1] if namespace else node_id
    role = _attr_text(attrs, "role") or ""
    details = _attr_text(attrs, "details") or ""
    return (
        section,
        computation,
        class_name,
        operation,
        output_text,
        input_texts,
        role,
        details,
    )


def _project_op(
    node_id: str,
    section: str,
    computation: str,
    class_name: str,
    operation: str,
    output_text: str,
    input_texts: list[list[str]],
    dims: dict[str, Any],
    instances: int,
    gpu_arch: dict[str, Any] | None,
    moe_top_k: int | None,
    moe_intermediate_size: int | None,
    role: str = "",
    details: str = "",
) -> RooflineOp:
    base = RooflineOp(
        name=node_id,
        section=section,
        computation=computation,
        operation=operation,
        category=None,
        modeled=False,
        reason=None,
        output_elements=None,
        instances=instances,
    )
    _dims_only, dtype = _split_shape_text(output_text)
    bpe = _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None

    # Introspective boundaries: the modeling source tags each module with a role
    # ("attention"/"ffn") and the arch spec says whether the model is MoE, so we
    # cost the custom attention and routed-expert blocks without matching names.
    if role == "attention":
        att = _attention_boundary_cost(dims, bpe)
        if att is not None:
            category, maf_type, flops, bytes_moved = att
            try:
                oshape = resolve_shape(_dims_only, dims)
                base.output_elements = prod(oshape) if oshape else 1
            except DimResolutionError:
                base.output_elements = None
            return _finish_op(
                base, category, maf_type, flops, bytes_moved, dtype, gpu_arch
            )
    if role == "ffn" and _RT.moe_top_k and _looks_like_moe(class_name):
        moe = _moe_cost(dims, _RT.moe_top_k, _RT.moe_intermediate, bpe)
        if moe is not None:
            category, maf_type, flops, bytes_moved, output_elements = moe
            base.output_elements = output_elements
            return _finish_op(
                base, category, maf_type, flops, bytes_moved, dtype, gpu_arch
            )
    # A dense MLP that shares the MoE attribute but was not parsed (both classes
    # live outside the registry) is one opaque tile. Cost the SwiGLU at the dense
    # width over every token; the MoE formula above would charge the routed
    # subset instead, which is the wrong layer.
    if (
        role == "ffn"
        and class_name
        and not _looks_like_moe(class_name)
        and not _is_leaf_computation(computation)
    ):
        dense = _dense_ffn_cost(dims, bpe)
        if dense is not None:
            category, maf_type, flops, bytes_moved, output_elements = dense
            base.output_elements = output_elements
            return _finish_op(
                base, category, maf_type, flops, bytes_moved, dtype, gpu_arch
            )
    if _is_short_conv(computation, class_name):
        conv = _short_conv_cost(dims, _dims_only, bpe)
        if conv is not None:
            category, maf_type, flops, bytes_moved, output_elements = conv
            base.output_elements = output_elements
            return _finish_op(
                base, category, maf_type, flops, bytes_moved, dtype, gpu_arch
            )

    engine_cost = _engine_boundary_cost(
        class_name, dims, moe_top_k, moe_intermediate_size, bpe
    )
    if engine_cost is not None:
        category, maf_type, flops, bytes_moved, output_elements = engine_cost
        base.output_elements = output_elements
        return _finish_op(base, category, maf_type, flops, bytes_moved, dtype, gpu_arch)

    try:
        output_shape = resolve_shape(_dims_only, dims)
        base.output_elements = prod(output_shape) if output_shape else 1
    except DimResolutionError as exc:
        base.reason = f"unresolved output dim {exc}"
        return base

    try:
        input_shapes = [
            resolve_shape(shape, dims) if shape else None for shape in input_texts
        ]
        cost = _dispatch_op_cost(computation, input_shapes, output_shape, bpe, details)
    except DimResolutionError as exc:
        base.reason = f"unresolved input dim {exc}"
        return base

    if cost is None:
        base.reason = f"no PerfModel mapping for computation={computation!r}"
        return base

    category, maf_type, flops, bytes_moved = cost
    return _finish_op(base, category, maf_type, flops, bytes_moved, dtype, gpu_arch)


def _finish_op(
    base: RooflineOp,
    category: str,
    maf_type: str,
    flops: float,
    bytes_moved: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> RooflineOp:
    base.category = category
    base.modeled = True
    base.flops = flops
    base.bytes = bytes_moved

    precision = torch_dtype_map(dtype) if dtype else None
    if precision is not None:
        base.compute_spec = f"{maf_type}_{precision}"
    if gpu_arch is not None and base.compute_spec is not None and flops:
        peak_tflops = (gpu_arch.get("max_achievable_tflops") or {}).get(
            base.compute_spec
        )
        mem_bw_gbps = gpu_arch.get("mem_bw_gbps")
        gflops = flops / 1e9
        roofline_time_us, roofline_bound = compute_roofline_time_and_bound(
            gflops, bytes_moved, peak_tflops, mem_bw_gbps
        )
        base.roofline_time_us = roofline_time_us
        base.roofline_bound = roofline_bound
    return base


def _engine_boundary_cost(
    class_name: str,
    dims: dict[str, Any],
    moe_top_k: int | None,
    moe_intermediate_size: int | None,
    bpe: float | None,
) -> tuple[str, str, float, float | None, int] | None:
    """``(category, maf_type, flops, bytes, output_elements)`` for a vLLM/SGLang
    boundary shape_inference has no rule for, or ``None`` when *class_name*
    names neither one (see the module docstring's approximations)."""
    if any(marker in class_name for marker in _ENGINE_HEAD_CLASS_MARKERS):
        return _engine_head_cost(dims, bpe)
    if any(marker in class_name for marker in _ENGINE_MOE_CLASS_MARKERS):
        return _moe_cost(dims, moe_top_k, moe_intermediate_size, bpe)
    return None


def _engine_head_cost(
    dims: dict[str, Any], bpe: float | None
) -> tuple[str, str, float, float | None, int] | None:
    """GEMM cost of the vocab projection a ``ParallelLMHead`` performs via its
    pluggable ``quant_method`` (never its own ``forward``, which only raises)."""
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        vocab = resolve_dim(dims[Symbol.VOCAB.value], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return None
    vocab = _shard(vocab, _RT.tp)  # ParallelLMHead column-shards the vocab across TP
    flops = GEMM.flops_func(tokens, vocab, hidden, bias=False)
    bytes_moved = GEMM.bytes_func(tokens, vocab, hidden, False, bpe, bpe, bpe, bpe)
    return "GEMM", "matrix", flops, bytes_moved, tokens * vocab


class _RuntimeIntrospection:
    """Per-run values introspected from the ArchitectureSpec (config.json +
    modeling source) so MoE / attention / quantization costs need no hardcoded
    model names. Populated by :func:`project_roofline`."""

    __slots__ = (
        "weight_bpe",
        "moe_top_k",
        "moe_intermediate",
        "moe_num_experts",
        "ep",
        "attn_heads",
        "attn_kv_heads",
        "attn_head_dim",
        "hidden_size",
        "d_rel",
        "rel_extent",
        "sliding_window",
        "n_local_layers",
        "n_layers",
        "tp",
        "ep",
        "sconv_kernel",
        "dense_intermediate",
        "mtp_layers",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.weight_bpe = None
        self.moe_top_k = None
        self.moe_intermediate = None
        self.moe_num_experts = None
        self.ep = 1
        self.attn_heads = None
        self.attn_kv_heads = None
        self.attn_head_dim = None
        self.hidden_size = None
        self.d_rel = None
        self.rel_extent = None
        self.sliding_window = None
        self.n_local_layers = None
        self.n_layers = None
        self.tp = 1
        self.ep = 1
        self.sconv_kernel = None
        self.dense_intermediate = None
        self.mtp_layers = None


_RT = _RuntimeIntrospection()


def _introspect_weight_bpe(quant):
    """Bytes/element of the quantized WEIGHT storage, from a HF
    quantization_config. None when weights are unquantized."""
    if not isinstance(quant, dict):
        return None
    try:
        from TraceLens.ModelUtils.shape_inference import _quant_storage_dtype

        wd = _quant_storage_dtype(quant)
    except Exception:
        wd = None
    if not wd:
        return None
    w = wd.lower()
    if "fp4" in w or "float4" in w or "mxfp4" in w or "e2m1" in w:
        return 0.53125  # 4-bit element + 1-byte e8m0 scale per 32 (MXFP4)
    if "fp8" in w or "float8" in w or "int8" in w:
        return 1.0
    if "int4" in w or "nf4" in w:
        return 0.5
    return None


def _populate_introspection(spec):
    _RT.reset()
    cfg = getattr(spec, "raw_config", None) or {}
    tcfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    _RT.weight_bpe = _introspect_weight_bpe(cfg.get("quantization_config"))
    _RT.moe_top_k = spec.num_experts_per_tok
    _RT.moe_intermediate = spec.moe_intermediate_size or spec.intermediate_size
    _RT.moe_num_experts = spec.num_experts
    layout = getattr(spec, "parallel_layout", None)
    _RT.ep = getattr(getattr(layout, "degree", None), "ep", 1) or 1
    _RT.attn_heads = spec.num_attention_heads
    _RT.attn_kv_heads = spec.num_key_value_heads or spec.num_attention_heads
    if spec.head_dim:
        _RT.attn_head_dim = spec.head_dim
    elif spec.hidden_size and spec.num_attention_heads:
        _RT.attn_head_dim = spec.hidden_size // spec.num_attention_heads
    _RT.hidden_size = spec.hidden_size
    # Relative-attention dims (models that add a learned per-position "r" stream
    # on top of SDPA, e.g. a d_rel-wide r-projection + a relative bias over a
    # bounded extent). Read generically from config; absent -> no rel term.
    d_rel = tcfg.get("d_rel")
    _RT.d_rel = d_rel if isinstance(d_rel, int) and d_rel > 0 else None
    rel_extent = tcfg.get("rel_extent")
    _RT.rel_extent = (
        rel_extent if isinstance(rel_extent, int) and rel_extent > 0 else None
    )
    sw = tcfg.get("sliding_window_size") or tcfg.get("sliding_window")
    _RT.sliding_window = sw if isinstance(sw, int) and sw > 0 else None
    lli = tcfg.get("local_layer_ids")
    _RT.n_local_layers = len(lli) if isinstance(lli, list) else None
    _RT.n_layers = spec.num_hidden_layers
    # Per-GPU share under a tensor-/expert-parallel split. The role-based
    # analytical costs (attention, projections, MoE) read config dims directly,
    # so they must divide the sharded dims themselves -- the localized node
    # shapes a GEMM leaf would carry do not reach these config-driven formulas.
    layout = getattr(spec, "parallel_layout", None)
    degree = getattr(layout, "degree", None)
    tp = getattr(degree, "tp", 1)
    ep = getattr(degree, "ep", 1)
    _RT.tp = tp if isinstance(tp, int) and tp > 0 else 1
    _RT.ep = ep if isinstance(ep, int) and ep > 0 else 1
    _RT.sconv_kernel = _config_int(
        tcfg, "sconv_kernel_size", "conv_kernel_size", "short_conv_kernel_size"
    )
    # The dense MLP's width, when the checkpoint distinguishes it from the
    # routed experts' width. Absent, the model's intermediate size is the
    # dense one (a non-MoE model has nothing else).
    _RT.dense_intermediate = _config_int(tcfg, "dense_intermediate_size") or (
        spec.intermediate_size
    )
    mtp_cfg = tcfg.get("mtp_config") if isinstance(tcfg.get("mtp_config"), dict) else {}
    _RT.mtp_layers = _config_int(
        mtp_cfg, "num_nextn_predict_layers", "num_mtp_layers"
    ) or _config_int(tcfg, "num_mtp_layers", "num_nextn_predict_layers")


def _config_int(config: dict[str, Any], *keys: str) -> int | None:
    """The first positive int among *keys* in *config*, else ``None``."""
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


def _activation_bpe(spec: ArchitectureSpec) -> tuple[float | None, str | None]:
    """``(bytes per element, dtype name)`` from the checkpoint's torch dtype."""
    cfg = getattr(spec, "raw_config", None) or {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    raw = text.get("torch_dtype") or cfg.get("torch_dtype")
    if not isinstance(raw, str):
        return 2.0, "bfloat16"
    mapped = torch_dtype_map(raw)
    return _BYTES_PER_ELEMENT.get(mapped), raw


def _looks_like_moe(class_name: str) -> bool:
    return "moe" in (class_name or "").lower()


def _is_short_conv(computation: str, class_name: str) -> bool:
    """A depthwise 1D short / causal conv, not a spatial Conv2d/Conv3d."""
    text = f"{computation} {class_name}".lower().replace("_", "")
    if "conv2d" in text or "conv3d" in text:
        return False
    return any(
        token in text for token in ("shortconv", "sconv", "causalconv", "conv1d")
    )


def _is_leaf_computation(computation: str) -> bool:
    """True when *computation* is an op the PerfModel dispatch already costs.

    An opaque module tile is named for its class (``DenseMLP``); costing that
    tile analytically must not also fire on a ``Linear`` or a norm that happens
    to carry an FFN role.
    """
    if computation in _GEMM_NN_MODULE_COMPUTATIONS:
        return True
    label = computation.lower()
    if (
        label
        in _GEMM_FUNCTIONAL_LABELS
        | _UNARY_POINTWISE_COMPUTATIONS
        | (_BINARY_POINTWISE_COMPUTATIONS)
        | _REDUCTION_COMPUTATIONS
    ):
        return True
    if computation in (_SDPA_COMPUTATION, _SOFTMAX_COMPUTATION):
        return True
    if _classify_normalization(computation) is not None:
        return True
    return _is_short_conv(computation, "")


def _shard(n: int, degree: int) -> int:
    """One GPU's share of *n* under a *degree*-way split (never below 1)."""
    return max(1, n // degree) if degree and degree > 1 else n


def _attention_boundary_cost(dims, bpe):
    """Roofline cost of ONE attention layer, introspected from the arch spec
    (query/kv heads, head dim, sliding window, local/global layer mix). KV length
    is averaged across the local (windowed) and global layers so the per-layer
    cost times the layer-repeat count reproduces the true aggregate."""
    if not (_RT.attn_heads and _RT.attn_head_dim):
        return None
    try:
        b = resolve_dim(dims["B"], dims)
        s = resolve_dim(dims["S"], dims)
    except (KeyError, DimResolutionError):
        return None
    past = 0
    for key in ("past_len", "past_length", "cache_len", "kv_len"):
        if key in dims:
            try:
                past = resolve_dim(dims[key], dims)
                break
            except DimResolutionError:
                pass
    ctx = past + s
    # Attention heads are column-sharded across TP; one GPU owns its share (KV
    # heads too, the way vLLM replicates/shards GQA by the TP degree).
    h_q = _shard(_RT.attn_heads, _RT.tp)
    h_kv = _shard(_RT.attn_kv_heads or _RT.attn_heads, _RT.tp)
    d_h = _RT.attn_head_dim
    n_layers = _RT.n_layers or 1
    if _RT.n_local_layers is not None:
        n_local = _RT.n_local_layers
    else:
        n_local = n_layers if _RT.sliding_window else 0
    n_global = max(n_layers - n_local, 0)
    kv_local = min(ctx, _RT.sliding_window) if _RT.sliding_window else ctx
    eff_n_kv = (n_local * kv_local + n_global * ctx) / n_layers if n_layers else ctx
    causal = False  # decode / windowed cross-attention: N_Q != N_KV
    flops = SDPA.flops_func(b, s, h_q, eff_n_kv, h_kv, d_h, d_h, causal)
    bytes_moved = (
        SDPA.bytes_func(b, s, h_q, eff_n_kv, h_kv, d_h, d_h, causal, bpe)
        if bpe is not None
        else None
    )
    return "Attention", "matrix", flops, bytes_moved


def _resolve_past(dims: dict[str, Any]) -> int:
    for key in ("past_len", "past_length", "cache_len", "kv_len"):
        if key in dims:
            try:
                return resolve_dim(dims[key], dims)
            except DimResolutionError:
                pass
    return 0


def _attention_extra_ops(
    section: str,
    node_id: str,
    instances: int,
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """The projection GEMMs (packed QKV + O) and, when the arch declares one, the
    relative-attention term (``r``-projection + relative-bias add) for a FUSED
    attention leaf.

    :func:`_attention_boundary_cost` costs only the SDPA core (QK^T + PV); a
    fused attention module (custom packed ``*ParallelLinear`` QKVR, e.g.
    vLLM/SGLang model-specific attention) packs its projections inside one opaque
    block, so they appear as NO separate ``Linear`` nodes and would otherwise be
    uncosted. This path runs ONLY for that fused boundary (a transformers-native
    attention expands into its own q_proj/k_proj/... Linear leaves, which the
    GEMM path already costs -- so there is no double count). General: every width
    comes from the arch spec's own dims + quantized weight bpe, mirroring
    :func:`_moe_cost`; returns ``[]`` when those dims are unknown."""
    if not (_RT.attn_heads and _RT.attn_head_dim and _RT.hidden_size):
        return []
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
    except (KeyError, DimResolutionError):
        return []
    hidden = _RT.hidden_size
    # Column-sharded projections: one GPU owns hidden/TP of the fused QKV output
    # and the O-proj input (the ``o_proj`` row below keeps the full hidden on its
    # other axis -- a RowParallel reduce).
    h_q = _shard(_RT.attn_heads, _RT.tp)
    h_kv = _shard(_RT.attn_kv_heads or _RT.attn_heads, _RT.tp)
    d_h = _RT.attn_head_dim
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    q_out = h_q * d_h
    kv_out = h_kv * d_h
    ops: list[RooflineOp] = []

    def _gemm_row(suffix: str, n: int, k: int, out_elems: int) -> RooflineOp:
        op = RooflineOp(
            name=f"{node_id}.{suffix}",
            section=section,
            computation=suffix,
            operation="",
            category=None,
            modeled=False,
            reason=None,
            output_elements=out_elems,
            instances=instances,
        )
        flops = GEMM.flops_func(tokens, n, k, bias=False)
        byt = (
            GEMM.bytes_func(tokens, n, k, False, wbpe, wbpe, bpe, bpe)
            if bpe is not None
            else None
        )
        return _finish_op(op, "GEMM", "matrix", flops, byt, dtype, gpu_arch)

    # vLLM/SGLang pack Q, K, V into one column-parallel projection; O is separate.
    ops.append(
        _gemm_row("qkv_proj", q_out + 2 * kv_out, hidden, tokens * (q_out + 2 * kv_out))
    )
    ops.append(_gemm_row("o_proj", hidden, q_out, tokens * hidden))

    # K and V short convs live inside the fused attention leaf (the residual
    # short convs are sibling modules and show up as their own nodes). A leaf
    # has no conv node, so the depthwise GEMM is dropped unless it is added here.
    if _RT.sconv_kernel:
        kv_channels = h_kv * d_h
        for suffix in ("k_sconv", "v_sconv"):
            ops.append(
                _short_conv_op(
                    f"{node_id}.{suffix}",
                    section,
                    suffix,
                    instances,
                    tokens,
                    kv_channels,
                    bpe,
                    dtype,
                    gpu_arch,
                )
            )

    # Relative-attention: a d_rel-wide r-projection GEMM plus a relative-bias add
    # over the attended span (bounded by rel_extent). Costed as its own rows so it
    # is visible rather than folded into the SDPA core.
    if _RT.d_rel:
        r_out = h_q * _RT.d_rel
        ops.append(_gemm_row("rel_r_proj", r_out, hidden, tokens * r_out))
        try:
            ctx = _resolve_past(dims) + resolve_dim(dims["S"], dims)
        except (KeyError, DimResolutionError):
            ctx = 0
        span = min(ctx, _RT.rel_extent) if _RT.rel_extent else ctx
        if span:
            bias_elems = tokens * h_q * span
            bias = RooflineOp(
                name=f"{node_id}.rel_bias",
                section=section,
                computation="rel_bias",
                operation="",
                category=None,
                modeled=False,
                reason=None,
                output_elements=bias_elems,
                instances=instances,
            )
            flops = BinaryElementwise.flops_func(bias_elems)
            byt = (
                BinaryElementwise.bytes_func(
                    bias_elems, bias_elems, bias_elems, bpe, bpe, bpe
                )
                if bpe is not None
                else None
            )
            ops.append(
                _finish_op(
                    bias, "BinaryElementwise", "vector", flops, byt, dtype, gpu_arch
                )
            )
    return ops


def _short_conv_cost(
    dims: dict[str, Any],
    shape_dims: list[str],
    bpe: float | None,
) -> tuple[str, str, float, float | None, int] | None:
    """Depthwise causal conv: ``2 * tokens * channels * kernel``.

    Channels come from the node's own trailing dim when it resolves, else the
    per-GPU hidden width. ``None`` when the config declares no kernel.
    """
    kernel = _RT.sconv_kernel
    if not kernel:
        return None
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
    except (KeyError, DimResolutionError):
        return None
    channels: int | None = None
    if shape_dims:
        try:
            shape = resolve_shape(shape_dims, dims)
        except DimResolutionError:
            shape = ()
        if shape:
            channels = shape[-1]
            tokens = prod(shape[:-1]) if len(shape) > 1 else tokens
    if not channels:
        if not _RT.hidden_size:
            return None
        channels = _shard(_RT.hidden_size, _RT.tp)
    flops = 2.0 * tokens * channels * kernel
    bytes_moved = None
    if bpe is not None:
        bytes_moved = (2 * tokens * channels + channels * kernel) * bpe
    return "Conv1d", "matrix", flops, bytes_moved, tokens * channels


def _short_conv_op(
    name: str,
    section: str,
    computation: str,
    instances: int,
    tokens: int,
    channels: int,
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> RooflineOp:
    kernel = _RT.sconv_kernel or 1
    op = RooflineOp(
        name=name,
        section=section,
        computation=computation,
        operation="",
        category=None,
        modeled=False,
        reason=None,
        output_elements=tokens * channels,
        instances=instances,
    )
    flops = 2.0 * tokens * channels * kernel
    bytes_moved = None
    if bpe is not None:
        bytes_moved = (2 * tokens * channels + channels * kernel) * bpe
    return _finish_op(op, "Conv1d", "matrix", flops, bytes_moved, dtype, gpu_arch)


def _dense_ffn_cost(
    dims: dict[str, Any], bpe: float | None
) -> tuple[str, str, float, float | None, int] | None:
    """SwiGLU GEMMs of a dense MLP: gate_up (2·intermediate) + down, every token.

    Not the routed-expert formula. Expert-parallel does not apply (there is one
    MLP, not a routed set); tensor-parallel column-shards the intermediate width.
    """
    intermediate = _RT.dense_intermediate
    if not intermediate:
        return None
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return None
    intermediate = _shard(intermediate, _RT.tp)
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    gate_up_flops = GEMM.flops_func(tokens, 2 * intermediate, hidden, bias=False)
    gate_up_bytes = GEMM.bytes_func(
        tokens, 2 * intermediate, hidden, False, wbpe, wbpe, bpe, bpe
    )
    down_flops = GEMM.flops_func(tokens, hidden, intermediate, bias=False)
    down_bytes = GEMM.bytes_func(
        tokens, hidden, intermediate, False, wbpe, wbpe, bpe, bpe
    )
    flops = gate_up_flops + down_flops
    bytes_moved = None if bpe is None else gate_up_bytes + down_bytes
    return "GEMM", "matrix", flops, bytes_moved, tokens * hidden


def _analytical_op(
    name: str,
    section: str,
    computation: str,
    instances: int,
    category: str,
    maf_type: str,
    flops: float,
    bytes_moved: float | None,
    output_elements: int,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> RooflineOp:
    op = RooflineOp(
        name=name,
        section=section,
        computation=computation,
        operation="",
        category=None,
        modeled=False,
        reason=None,
        output_elements=output_elements,
        instances=instances,
    )
    return _finish_op(op, category, maf_type, flops, bytes_moved, dtype, gpu_arch)


def _mtp_stack_ops(
    dims: dict[str, Any],
    num_predicted_tokens: int,
    gpu_arch: dict[str, Any] | None,
    activation: tuple[float | None, str | None],
) -> list[RooflineOp]:
    """One draft depth per predicted token, as its own stack.

    Capped at the depth the checkpoint declares (``mtp_config`` /
    ``num_mtp_layers``). Each depth is the draft block: dual RMSNorm, a
    replicated 2H→H input projection, one attention (with its packed
    projections), a dense MLP — not the routed experts — and, when the arch
    has a short conv, the residual convs. The shared LM head runs once per
    draft token.
    """
    if num_predicted_tokens <= 0:
        return []
    depth = _RT.mtp_layers
    count = min(num_predicted_tokens, depth) if depth else num_predicted_tokens
    if count <= 0 or not _RT.hidden_size:
        return []
    bpe, dtype = activation
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return []
    ops: list[RooflineOp] = []
    for index in range(count):
        section = f"mtp/{index}"
        node_id = f"mtp/{index}"
        # Two RMSNorms (hidden stream and embedding stream) before the fuse.
        norm_elems = tokens * hidden
        norm_flops = _rmsnorm_flops(norm_elems, hidden, True)
        norm_bytes = (
            _rmsnorm_bytes(norm_elems, hidden, True, bpe, bpe)
            if bpe is not None
            else None
        )
        for suffix in ("hidden_norm", "embed_norm"):
            ops.append(
                _analytical_op(
                    f"{node_id}.{suffix}",
                    section,
                    suffix,
                    1,
                    "Normalization (RMSNorm)",
                    "vector",
                    norm_flops,
                    norm_bytes,
                    norm_elems,
                    dtype,
                    gpu_arch,
                )
            )
        # Replicated 2H → H projection of the concatenated normed streams.
        proj_flops = GEMM.flops_func(tokens, hidden, 2 * hidden, bias=False)
        proj_bytes = (
            GEMM.bytes_func(tokens, hidden, 2 * hidden, False, bpe, bpe, bpe, bpe)
            if bpe is not None
            else None
        )
        ops.append(
            _analytical_op(
                f"{node_id}.input_proj",
                section,
                "input_proj",
                1,
                "GEMM",
                "matrix",
                proj_flops,
                proj_bytes,
                tokens * hidden,
                dtype,
                gpu_arch,
            )
        )
        attention = _attention_boundary_cost(dims, bpe)
        if attention is not None:
            category, maf_type, flops, bytes_moved = attention
            ops.append(
                _analytical_op(
                    f"{node_id}.attn",
                    section,
                    "attn",
                    1,
                    category,
                    maf_type,
                    flops,
                    bytes_moved,
                    tokens * hidden,
                    dtype,
                    gpu_arch,
                )
            )
            ops.extend(
                _attention_extra_ops(
                    section, f"{node_id}.attn", 1, dims, bpe, dtype, gpu_arch
                )
            )
        dense = _dense_ffn_cost(dims, bpe)
        if dense is not None:
            category, maf_type, flops, bytes_moved, out_elems = dense
            ops.append(
                _analytical_op(
                    f"{node_id}.mlp",
                    section,
                    "mlp",
                    1,
                    category,
                    maf_type,
                    flops,
                    bytes_moved,
                    out_elems,
                    dtype,
                    gpu_arch,
                )
            )
        if _RT.sconv_kernel:
            channels = _shard(hidden, _RT.tp)
            for suffix in ("attn_sconv", "mlp_sconv"):
                ops.append(
                    _short_conv_op(
                        f"{node_id}.{suffix}",
                        section,
                        suffix,
                        1,
                        tokens,
                        channels,
                        bpe,
                        dtype,
                        gpu_arch,
                    )
                )
        head = _engine_head_cost(dims, bpe)
        if head is not None:
            category, maf_type, flops, bytes_moved, out_elems = head
            ops.append(
                _analytical_op(
                    f"{node_id}.lm_head",
                    section,
                    "lm_head",
                    1,
                    category,
                    maf_type,
                    flops,
                    bytes_moved,
                    out_elems,
                    dtype,
                    gpu_arch,
                )
            )
    return ops


def _moe_cost(
    dims: dict[str, Any],
    top_k: int | None,
    intermediate_size: int | None,
    bpe: float | None,
) -> tuple[str, str, float, float | None, int] | None:
    """GroupedGEMM cost of the gate_up_proj + down_proj of the SELECTED experts,
    plus the EP dispatch/combine all-to-all when expert-parallel.

    ``top_k``/``intermediate_size`` come from the architecture's own MoE
    config (``spec.num_experts_per_tok``, ``spec.moe_intermediate_size``), not
    ``dims[Symbol.INTERMEDIATE]`` -- that key prefers a model's DENSE
    ``intermediate_size`` (shared-expert / non-MoE layers) when both are
    declared, which is the wrong width for the routed experts this costs.

    The expert GEMMs are costed with :class:`GroupedGemm` over
    ``G = min(num_experts, active_tokens)`` distinct expert weight matrices --
    the weights actually streamed from HBM for this token count -- rather than a
    single shared weight. FLOPs are unchanged (only the selected token-expert
    pairs compute); the weight-byte term now scales with the experts touched, so
    the memory-bound/compute-bound crossover tracks the real prefill-vs-decode
    behaviour instead of understating decode weight traffic by ~``G``x. When
    ``num_experts`` is unknown ``G`` falls back to 1 (the prior single-weight
    cost).

    When ``_RT.ep > 1`` the dispatch and combine all-to-all that move token
    hidden states between expert-parallel ranks are added as pure data movement
    (:class:`MoEComm`, FLOPs-free). This is the MoE op's own intrinsic
    communication, distinct from the ``@comm:`` TP/PP collectives the projection
    leaves to TreePerf.
    """
    if not top_k or not intermediate_size:
        return None
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return None
    # Per-GPU share: expert-parallel splits the routed rows; tensor-parallel
    # column-shards each expert's intermediate width. GroupedGemm then counts
    # the distinct expert weights this GPU actually streams.
    intermediate_size = _shard(intermediate_size, _RT.tp)
    active_tokens = _shard(tokens * top_k, _RT.ep)
    experts_here = (
        _shard(_RT.moe_num_experts, _RT.ep) if _RT.moe_num_experts else 0
    )
    groups = min(experts_here, active_tokens) if experts_here else 1
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    # gate_up_proj: [active_tokens, hidden] x per-expert [hidden, 2*intermediate].
    gate_up_flops = GroupedGemm.flops_func(active_tokens, hidden, 2 * intermediate_size)
    # down_proj: [active_tokens, intermediate] x per-expert [intermediate, hidden].
    down_flops = GroupedGemm.flops_func(active_tokens, intermediate_size, hidden)
    flops = gate_up_flops + down_flops
    bytes_moved: float | None = None
    if bpe is not None:
        gate_up_bytes = GroupedGemm.bytes_func(
            active_tokens, hidden, 2 * intermediate_size, groups, wbpe, bpe
        )
        down_bytes = GroupedGemm.bytes_func(
            active_tokens, intermediate_size, hidden, groups, wbpe, bpe
        )
        bytes_moved = gate_up_bytes + down_bytes
        if _RT.ep > 1:
            # Dispatch sends each token's hidden state to its top_k expert ranks;
            # combine gathers the expert outputs back (MoEComm: num_tokens * hidden
            # * bpe each). Token activations move at the activation dtype, not wbpe.
            bytes_moved += 2 * active_tokens * hidden * bpe
    return "MoE", "matrix", flops, bytes_moved, tokens * hidden


def _split_shape_text(text: str) -> tuple[list[str], str | None]:
    """Split a drawn ``"[B, S, 4096] bfloat16"`` attr into (dims, dtype).

    The dims list itself holds comma-separated, space-containing dims (``[B,
    S, 4096]``), so the dtype suffix has to split on the closing bracket, not
    on the first space.
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return parse_shape_dims(stripped), None
    dims_part, bracket, dtype_part = stripped.partition("]")
    return parse_shape_dims(dims_part + bracket), dtype_part.strip() or None


def _dispatch_op_cost(
    computation: str,
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
    details: str = "",
) -> tuple[str, str, float, float | None] | None:
    """Return ``(category, maf_type, flops, bytes)`` for an op, or ``None`` if unmodeled."""
    if computation in _GEMM_NN_MODULE_COMPUTATIONS:
        return _gemm_cost_linear(input_shapes, output_shape, bpe)
    if computation.lower() in _GEMM_FUNCTIONAL_LABELS:
        return _gemm_cost_matmul(input_shapes, output_shape, bpe)
    if computation in _CONV_COMPUTATIONS:
        return _conv_cost(details, input_shapes, output_shape, bpe)
    if computation == _SDPA_COMPUTATION:
        return _sdpa_cost(input_shapes, bpe)
    normalization_family = _classify_normalization(computation)
    if normalization_family is not None:
        kind, has_bias, channel_first = normalization_family
        return _normalization_cost(
            computation, kind, has_bias, channel_first, output_shape, bpe
        )
    if computation == _SOFTMAX_COMPUTATION:
        return _softmax_cost(output_shape, bpe)
    label = computation.lower()
    if label in _REDUCTION_COMPUTATIONS:
        return _reduce_cost(input_shapes, output_shape, bpe)
    if label in _BINARY_POINTWISE_COMPUTATIONS:
        return _binary_elementwise_cost(input_shapes, output_shape, bpe)
    if label in _UNARY_POINTWISE_COMPUTATIONS:
        return _unary_elementwise_cost(output_shape, bpe)
    return None


def _gemm_cost_linear(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not input_shapes or not input_shapes[0] or not output_shape:
        return None
    input_shape = input_shapes[0]
    k = input_shape[-1]
    n = output_shape[-1]
    m = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    flops = GEMM.flops_func(m, n, k, bias=False)
    bytes_moved = GEMM.bytes_func(m, n, k, False, bpe, bpe, bpe, bpe)
    return "GEMM", "matrix", flops, bytes_moved


def _gemm_cost_matmul(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if len(input_shapes) < 2:
        return None
    a_shape, b_shape = input_shapes[0], input_shapes[1]
    if not a_shape or not b_shape or not output_shape:
        return None
    k = a_shape[-1]
    n = b_shape[-1]
    m = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    flops = GEMM.flops_func(m, n, k, bias=False)
    bytes_moved = GEMM.bytes_func(m, n, k, False, bpe, bpe, bpe, bpe)
    return "GEMM", "matrix", flops, bytes_moved


_CONV_WEIGHT_RE = re.compile(r"conv_weight=\[([0-9,\s]+)\]")
_CONV_FLAG_RE = re.compile(r"conv_(bias|transposed)=([01])")


def _parse_conv_details(details: str) -> tuple[tuple[int, ...], bool, bool] | None:
    """``(weight_shape, bias, transposed)`` from a node's ``details`` attr, or
    ``None`` when the convolution geometry was not threaded onto the node.

    The meta-device pass stamps ``conv_weight=[C_out, C_in/groups, *kernel]``,
    ``conv_bias=0|1``, and ``conv_transposed=0|1`` (see ``meta_trace`` /
    ``_CONV_COMPUTATIONS``)."""
    if not details:
        return None
    match = _CONV_WEIGHT_RE.search(details)
    if not match:
        return None
    try:
        weight = tuple(int(tok) for tok in match.group(1).split(",") if tok.strip())
    except ValueError:
        return None
    if not weight:
        return None
    flags = {k: v == "1" for k, v in _CONV_FLAG_RE.findall(details)}
    return weight, flags.get("bias", False), flags.get("transposed", False)


def _conv_cost(
    details: str,
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    """CONV FLOPs/bytes from the input/output shapes plus the weight geometry on
    the node's ``details`` attr. The strides/padding/dilation that `nn.Conv*`
    uses only determine the output shape, which is already resolved here, so the
    static ``CONV.flops_func``/``bytes_func`` need only the three shapes plus the
    bias/transposed flags."""
    parsed = _parse_conv_details(details)
    if parsed is None or not input_shapes or not input_shapes[0] or not output_shape:
        return None
    weight_shape, bias, transposed = parsed
    input_shape = input_shapes[0]
    flops = CONV.flops_func(input_shape, weight_shape, output_shape, bias, transposed)
    bytes_moved = (
        CONV.bytes_func(input_shape, weight_shape, output_shape, bias, bpe)
        if bpe is not None
        else None
    )
    return "CONV", "matrix", flops, bytes_moved


def conv_ops_from_meta(
    meta_convs: list[dict[str, Any]] | None,
    *,
    dtype: str = "bfloat16",
    gpu_arch: dict[str, Any] | None = None,
) -> list[RooflineOp]:
    """Roofline ops for every convolution a meta-device forward captured.

    Each entry is one live ``_ConvNd`` instance from
    :func:`TraceLens.ModelUtils.meta_trace.trace_meta_conv_ops`, carrying the
    concrete ``[B, C, H, W]`` in/out shapes and weight geometry CONV needs. The
    graph draws one conv per stage; each live instance here carries the geometry
    to cost that instance, and the CLI uses these rows in place of the graph's
    conv rows so the repeated stage is not counted twice."""
    bpe = _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
    ops: list[RooflineOp] = []
    for conv in meta_convs or []:
        out_shape = tuple(conv.get("output_shape") or ())
        in_shape = tuple(conv["input_shape"]) if conv.get("input_shape") else None
        weight_shape = tuple(conv.get("weight_shape") or ())
        if not out_shape or in_shape is None or not weight_shape:
            continue
        path = str(conv.get("path") or "")
        base = RooflineOp(
            name=path,
            section=path.split(".")[-1] or path,
            computation="Conv",
            operation="",
            category=None,
            modeled=False,
            reason=None,
            output_elements=prod(out_shape),
            instances=1,
        )
        bias = bool(conv.get("bias"))
        transposed = bool(conv.get("transposed"))
        flops = CONV.flops_func(in_shape, weight_shape, out_shape, bias, transposed)
        bytes_moved = (
            CONV.bytes_func(in_shape, weight_shape, out_shape, bias, bpe)
            if bpe is not None
            else None
        )
        ops.append(
            _finish_op(base, "CONV", "matrix", flops, bytes_moved, dtype, gpu_arch)
        )
    return ops


def _sdpa_cost(
    input_shapes: list[tuple[int, ...] | None],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if len(input_shapes) < 3:
        return None
    q_shape, k_shape, v_shape = input_shapes[0], input_shapes[1], input_shapes[2]
    if not q_shape or not k_shape or not v_shape:
        return None
    if len(q_shape) < 4 or len(k_shape) < 4 or len(v_shape) < 4:
        return None
    # [B, heads, seq, head_dim], per the graph's own SDPA shape rule.
    b, h_q, n_q, d_h_qk = q_shape[-4:]
    _, h_kv, n_kv, _ = k_shape[-4:]
    d_h_v = v_shape[-1]
    causal = n_q == n_kv
    flops = SDPA.flops_func(b, n_q, h_q, n_kv, h_kv, d_h_qk, d_h_v, causal)
    bytes_moved = (
        SDPA.bytes_func(b, n_q, h_q, n_kv, h_kv, d_h_qk, d_h_v, causal, bpe)
        if bpe is not None
        else None
    )
    return "SDPA", "matrix", flops, bytes_moved


def _normalization_cost(
    computation: str,
    kind: str,
    has_bias: bool,
    channel_first: bool,
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    num_elems = prod(output_shape)
    if channel_first and len(output_shape) > 1:
        num_channels = output_shape[1]
    else:
        num_channels = output_shape[-1]
    if not num_channels:
        return None
    is_affine = True
    if kind == "rmsnorm":
        flops = _rmsnorm_flops(num_elems, num_channels, is_affine)
        bytes_moved = (
            _rmsnorm_bytes(num_elems, num_channels, is_affine, bpe, bpe)
            if bpe is not None
            else None
        )
    else:
        flops = Normalization.flops_func(
            has_bias, is_affine, False, num_elems, num_channels
        )
        bytes_moved = (
            Normalization.bytes_func(
                has_bias, is_affine, False, num_elems, num_channels, bpe, bpe
            )
            if bpe is not None
            else None
        )
    return f"Normalization ({computation})", "vector", flops, bytes_moved


def _rmsnorm_flops(num_elems: int, num_channels: int, is_affine: bool) -> float:
    """Mirrors RMSNorm.flops() in TraceLens.PerfModel.perf_model (no static
    flops_func exists there -- RMSNorm overrides the instance method directly)."""
    non_normalized_elems = num_elems / num_channels
    flops = non_normalized_elems * (2 * num_channels + 2)
    flops += num_elems * (2 if is_affine else 1)
    return flops


def _rmsnorm_bytes(
    num_elems: int, num_channels: int, is_affine: bool, bpe_in: float, bpe_out: float
) -> float:
    """Mirrors RMSNorm.bytes() in TraceLens.PerfModel.perf_model."""
    return (
        num_elems * bpe_in
        + num_elems * bpe_out
        + (num_channels * bpe_in if is_affine else 0)
    )


def _softmax_cost(
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    n = output_shape[-1]
    m = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    flops = Softmax.flops_func(m, n)
    bytes_moved = Softmax.bytes_func(m, n, bpe) if bpe is not None else None
    return "Softmax", "vector", flops, bytes_moved


def _unary_elementwise_cost(
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    nelems = prod(output_shape)
    flops = UnaryElementwise.flops_func(nelems)
    bytes_moved = (
        UnaryElementwise.bytes_func(nelems, bpe, bpe) if bpe is not None else None
    )
    return "UnaryElementwise", "vector", flops, bytes_moved


def _binary_elementwise_cost(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    nelems_out = prod(output_shape)
    # A missing operand is a scalar constant (e.g. "* 2", "+ eps") too small to
    # carry its own shape; one element is the honest size for such an operand.
    shape1 = input_shapes[0] if input_shapes else None
    shape2 = input_shapes[1] if len(input_shapes) > 1 else None
    nelems_in1 = prod(shape1) if shape1 else 1
    nelems_in2 = prod(shape2) if shape2 else 1
    flops = BinaryElementwise.flops_func(nelems_out)
    bytes_moved = (
        BinaryElementwise.bytes_func(nelems_in1, nelems_in2, nelems_out, bpe, bpe, bpe)
        if bpe is not None
        else None
    )
    return "BinaryElementwise", "vector", flops, bytes_moved


def _reduce_cost(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    if not input_shapes or not input_shapes[0]:
        return None
    input_shape = input_shapes[0]
    num_input_elems = prod(input_shape)
    num_output_elems = prod(output_shape) if output_shape else 1
    flops = Reduce.flops_func(num_input_elems, num_output_elems)
    bytes_moved = (
        Reduce.bytes_func(num_input_elems, num_output_elems, bpe, bpe)
        if bpe is not None
        else None
    )
    return "Reduce", "vector", flops, bytes_moved
