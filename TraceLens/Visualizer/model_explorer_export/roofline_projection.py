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

Op coverage is intentionally narrow: GEMM (``Linear``/``matmul``/``bmm``), SDPA,
the Normalization family, Softmax, generic elementwise/reduction ops, and MoE
expert routing -- exactly the leaf granularity
:func:`TraceLens.ModelUtils.build_roofline_basic_ops` stops AST expansion at,
plus the vLLM/SGLang engine boundaries (``ParallelLMHead``, ``FusedMoEFactory``)
that never expand at all (see "Known approximations" below). Anything else
(embeddings, convolutions -- CONV needs kernel/stride geometry the graph does
not carry) is reported as "unmodeled" rather than silently dropped, so the
projection's coverage is always visible instead of assumed.

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
  - MoE routing is costed as ``experts_per_tok`` GEMM passes per token (the
    gate_up_proj and down_proj of only the SELECTED experts), never
    ``num_experts`` passes -- the router picks a subset per token, so costing
    every expert for every token would overstate FLOPs by
    ``num_experts / experts_per_tok``. The router's own softmax/top-k
    selection and the shared-expert add are not separately costed (both are
    O(tokens), negligible next to the O(tokens*hidden*intermediate) expert
    GEMMs).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import prod
from typing import Any

from TraceLens.ModelUtils.extract import ArchitectureSpec
from TraceLens.ModelUtils.loader import build_roofline_basic_ops
from TraceLens.ModelUtils.shape_inference import ShapeContext, Symbol
from TraceLens.PerfModel.perf_model import (
    GEMM,
    SDPA,
    BinaryElementwise,
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

# A node's instance count is the innermost ``"<N>x_..."`` repeat-count
# namespace segment around it -- the merged graph's own per-variant tile count.
_REPEAT_NAMESPACE_SEGMENT = re.compile(r"^(\d+)x(?:_|$)")

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
) -> RooflineProjection:
    """Project a kernel-by-kernel roofline for *spec* at a concrete batch/seq length.

    Builds the same merged Model Explorer graph the visualizer draws, with
    expansion stopped at roofline-formula ops (:func:`build_roofline_basic_ops`)
    and, when ``spec.parallel_layout`` is set, the same ``--tp``/``--pp``/``--ep``
    split and per-GPU local shapes the visualizer would show for that layout.
    """
    dims = build_projection_dims(
        spec, batch_size=batch_size, seq_len=seq_len, extra_dims=extra_dims
    )
    _populate_introspection(spec)
    moe_top_k = spec.num_experts_per_tok
    moe_intermediate_size = spec.moe_intermediate_size or spec.intermediate_size
    payload = build_model_explorer_payload(
        spec, basic_ops=build_roofline_basic_ops(), include_shapes=True
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
        section, computation, class_name, operation, output_text, input_texts, role = (
            parsed
        )
        instances = _repeat_count(str(node.get("namespace") or ""))
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
    return result


def _repeat_count(namespace: str) -> int:
    """The innermost ``"<N>x_..."`` tile-repeat count in *namespace*, or 1."""
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
    return section, computation, class_name, operation, output_text, input_texts, role


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
    if role == "ffn" and _RT.moe_top_k and "moe" in class_name.lower():
        moe = _moe_cost(dims, _RT.moe_top_k, _RT.moe_intermediate, bpe)
        if moe is not None:
            category, maf_type, flops, bytes_moved, output_elements = moe
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
        cost = _dispatch_op_cost(computation, input_shapes, output_shape, bpe)
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
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.weight_bpe = None
        self.moe_top_k = None
        self.moe_intermediate = None
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
    _RT.rel_extent = rel_extent if isinstance(rel_extent, int) and rel_extent > 0 else None
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
                _finish_op(bias, "BinaryElementwise", "vector", flops, byt, dtype, gpu_arch)
            )
    return ops


def _moe_cost(
    dims: dict[str, Any],
    top_k: int | None,
    intermediate_size: int | None,
    bpe: float | None,
) -> tuple[str, str, float, float | None, int] | None:
    """GEMM cost of the gate_up_proj + down_proj of the SELECTED experts only.

    ``top_k``/``intermediate_size`` come from the architecture's own MoE
    config (``spec.num_experts_per_tok``, ``spec.moe_intermediate_size``), not
    ``dims[Symbol.INTERMEDIATE]`` -- that key prefers a model's DENSE
    ``intermediate_size`` (shared-expert / non-MoE layers) when both are
    declared, which is the wrong width for the routed experts this costs.
    """
    if not top_k or not intermediate_size:
        return None
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return None
    # Per-GPU share: expert-parallel splits the routed experts across EP GPUs
    # (each sees tokens*top_k/EP expert rows); tensor-parallel column-shards each
    # expert's intermediate width across TP.
    intermediate_size = _shard(intermediate_size, _RT.tp)
    active_tokens = _shard(tokens * top_k, _RT.ep)
    gate_up_flops = GEMM.flops_func(
        active_tokens, 2 * intermediate_size, hidden, bias=False
    )
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    gate_up_bytes = GEMM.bytes_func(
        active_tokens, 2 * intermediate_size, hidden, False, wbpe, wbpe, bpe, bpe
    )
    down_flops = GEMM.flops_func(active_tokens, hidden, intermediate_size, bias=False)
    down_bytes = GEMM.bytes_func(
        active_tokens, hidden, intermediate_size, False, wbpe, wbpe, bpe, bpe
    )
    flops = gate_up_flops + down_flops
    bytes_moved = None if bpe is None else gate_up_bytes + down_bytes
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
) -> tuple[str, str, float, float | None] | None:
    """Return ``(category, maf_type, flops, bytes)`` for an op, or ``None`` if unmodeled."""
    if computation in _GEMM_NN_MODULE_COMPUTATIONS:
        return _gemm_cost_linear(input_shapes, output_shape, bpe)
    if computation.lower() in _GEMM_FUNCTIONAL_LABELS:
        return _gemm_cost_matmul(input_shapes, output_shape, bpe)
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
