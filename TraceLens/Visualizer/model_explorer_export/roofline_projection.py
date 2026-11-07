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
``conv_kernel_size``) because that leaf has no conv node of its own. An
embedding lookup is a gather: zero FLOPs, and bytes for the gathered rows,
their write, and the index vector. Anything else (a conv with no threaded
geometry and no declared short-conv kernel) is reported as "unmodeled"
rather than silently dropped, so the projection's coverage is always visible
instead of assumed.

Convolutional models are handled separately. The graph expands a non-uniform
stage once (one conv node per stage, plus the stem and each stage's leading
downsampling conv). :func:`conv_ops_from_meta` still costs every live conv
instance a meta-device forward captures (see
:func:`TraceLens.ModelUtils.meta_trace.trace_meta_conv_ops`) -- each already
carries the concrete ``[B, C, H, W]`` shapes and weight geometry CONV needs --
and the CLI lets those supersede the graph's conv rows so a repeated stage is
not counted twice.

Known approximations, stated once here rather than scattered in comments:
  - A GEMM includes its bias add when the node details or the checkpoint
    flags ``attention_bias``, ``mlp_bias``, or ``lm_head_bias`` say so.
  - Every Normalization op is assumed affine; RMSNorm is assumed bias-free,
    every other norm is assumed to have a bias.
  - SDPA follows the checkpoint's causal flag. Decoder-only models are
    causal. The mask applies on cold prefill, when the query length is the
    whole key/value length; a warm cache keeps the unmasked FLOP count.
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
    all-to-all as FLOPs-free data movement (:class:`MoEComm`). The router's
    GEMM, its softmax, the combine, and a shared expert are their own rows
    when the graph did not already draw them. A dense MLP that
    shares the same attribute (the layers below a ``dense_mlp_idx`` threshold)
    is costed as a SwiGLU over every token at the dense intermediate width, not
    as that routed subset.
  - A depthwise short convolution packed inside a fused attention leaf is
    costed when the config declares its kernel (``sconv_kernel_size`` /
    ``conv_kernel_size``): ``2 * tokens * channels * kernel``. A graph conv
    node uses the threaded ``conv_weight`` geometry instead.
  - ``num_predicted_tokens`` adds a secondary MTP / speculative draft stack:
    one draft depth per predicted token (capped at the depth the config
    declares). The draft layer is the speculative class in the modeling
    source, and its feed-forward follows that layer, dense or MoE. With no
    such layer in the source, the stack falls back to one attention plus a
    dense MLP. Leaves the graph already drew are not emitted again; an
    attention core or expert GEMM the graph inlined away still is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field, replace
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
    IndexedCopy,
    Normalization,
    Reduce,
    Softmax,
    UnaryElementwise,
    aten_sqrt,
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
# slice/...) are deliberately left out of these sets -- they are usually free
# views or fused into a neighboring kernel, and PerfModel has no formula for
# them; they fall into "unmodeled" rather than being charged a cost they may
# not actually incur. Tensor indexing (``Indexing``, ``Index select``,
# ``aten::index``) is a gather and is costed with the embedding lookup.
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
    cost_source: str = "graph"
    """Where the cost came from: graph leaf, parsed source, config, or fallback."""
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
    notes: list[str] = field(default_factory=list)
    """Engine-source misses and other reasons a width fell back to the config."""

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
            "notes": list(self.notes),
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
        if _node_is_draft_container(node):
            # The synthetic stack, or the container's own drawn children, costs
            # this module. The container node itself is not a second copy.
            continue
        result.ops.append(op)
        _dims_only, dtype = _split_shape_text(output_text)
        bpe = _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
        if (
            role == "attention"
            and op.modeled
            and op.category == "Attention"
            and _attention_variants_cover(op.instances)
        ):
            result.ops.pop()
            result.ops.extend(
                _attention_variant_rows(
                    op, node_id, section, nodes, node, dims, bpe, dtype, gpu_arch
                )
            )
        elif (
            role == "attention"
            and op.modeled
            and op.category == "Attention"
            and not _attention_has_projection_leaves(nodes, node)
        ):
            # A fused attention leaf is costed at only its SDPA core. Its packed
            # projections, indexer, and norms live inside the opaque block.
            result.ops.extend(
                _attention_extra_ops(
                    section,
                    node_id,
                    instances,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                    owner_class=class_name,
                )
            )
        if (
            role in ("ffn", "moe")
            and op.modeled
            and op.category == "MoE"
            and not _is_leaf_computation(computation)
        ):
            result.ops.extend(
                _moe_extra_ops(
                    section,
                    node_id,
                    instances,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                    present=_moe_leaves_present(nodes, node),
                )
            )
        if (
            op.modeled
            and op.category == "GEMM"
            and computation in _GEMM_NN_MODULE_COMPUTATIONS
            and _RT.weight_bpe is not None
        ):
            result.ops.extend(
                _quant_side_ops(
                    node_id,
                    section,
                    instances,
                    input_texts,
                    output_text,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                )
            )
        elif (
            op.modeled and op.category in ("GEMM", "MoE") and _RT.weight_bpe is not None
        ):
            result.ops.extend(
                _formula_quant_rows(
                    node_id,
                    section,
                    instances,
                    op.category,
                    role,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                )
            )
    if num_predicted_tokens > 0:
        draft_nodes = _draft_graph_nodes(nodes, spec)
        result.ops.extend(
            _mtp_stack_ops(
                dims,
                num_predicted_tokens,
                gpu_arch,
                _activation_bpe(spec),
                spec=spec,
                drawn=draft_nodes or None,
            )
        )
    result.ops.extend(_persistent_state_ops(nodes, dims, gpu_arch))
    if _RT.engine_miss:
        result.notes.append(_RT.engine_miss)
    return result


def _persistent_state_ops(
    nodes: list[dict[str, Any]],
    dims: dict[str, Any],
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """Cost typed persistent-state boundaries recorded by the traced graph.

    A state read and a state write are independent transfers: a stateful
    forward can replace a state with a different shape or dtype.  This relies
    only on graph metadata and the tensor at each boundary, not model names,
    cache conventions, or a presumed read/write symmetry.
    """
    if gpu_arch is None or not gpu_arch.get("mem_bw_gbps"):
        return []
    state_ops: list[RooflineOp] = []
    for node in nodes:
        if _attr_text(node.get("attrs"), "synthetic"):
            continue
        output_text = _attr_text(node.get("attrs"), "output_shape")
        if not output_text:
            continue
        shape_text, dtype = _split_shape_text(output_text)
        bpe = _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
        if bpe is None:
            continue
        try:
            elements = prod(resolve_shape(shape_text, dims))
        except DimResolutionError:
            continue
        bytes_moved = elements * bpe
        roofline_time_us = bytes_moved / (gpu_arch["mem_bw_gbps"] * 1e9) * 1e6
        for direction, attr in (("read", "state_read"), ("write", "state_write")):
            for state_name in _attr_values(node.get("attrs"), attr):
                state_ops.append(
                    RooflineOp(
                        name=f"{node.get('id')}:persistent-state-{direction}:{state_name}",
                        section=str(node.get("namespace") or ""),
                        computation=f"persistent state {direction} ({state_name})",
                        operation="",
                        category="Persistent State",
                        modeled=True,
                        reason=None,
                        output_elements=elements,
                        instances=_repeat_count(str(node.get("namespace") or ""), node),
                        flops=0.0,
                        bytes=bytes_moved,
                        roofline_time_us=roofline_time_us,
                        roofline_bound="MEMORY_BOUND",
                    )
                )
    return state_ops


def _attention_has_projection_leaves(
    nodes: list[Any], attention_node: dict[str, Any]
) -> bool:
    """Whether a collapsed attention tile already exposes its projection GEMMs.

    Engine model files frequently contain both an ``Attention`` boundary and
    sibling ``QKVParallelLinear``/``RowParallelLinear`` leaves.  The boundary
    formula covers SDPA only; adding analytical QKV/O rows in that case would
    count the projections twice.  Truly opaque custom attention exposes neither
    leaf and continues through the analytical fallback.
    """
    namespace = str(attention_node.get("namespace") or "")
    if not namespace:
        return False
    for node in nodes:
        if not isinstance(node, dict) or node is attention_node:
            continue
        child_namespace = str(node.get("namespace") or "")
        if not (
            child_namespace == namespace or child_namespace.startswith(namespace + "/")
        ):
            continue
        class_name = _attr_text(node.get("attrs"), "class_name") or ""
        if _node_is_linear(class_name):
            return True
    return False


def _node_is_linear(class_name: str) -> bool:
    """A drawn module whose class is a linear, engine wrapper included."""
    from TraceLens.ModelUtils.ast_analyze import displays_as_linear

    return displays_as_linear("", class_name)


def _child_nodes(nodes: list[Any], parent: dict[str, Any]) -> list[dict[str, Any]]:
    namespace = str(parent.get("namespace") or "")
    if not namespace:
        return []
    children = []
    for node in nodes:
        if not isinstance(node, dict) or node is parent:
            continue
        child_namespace = str(node.get("namespace") or "")
        if child_namespace == namespace or child_namespace.startswith(namespace + "/"):
            children.append(node)
    return children


def _moe_leaves_present(nodes: list[Any], moe_node: dict[str, Any]) -> frozenset[str]:
    """Router and shared-expert leaves already drawn under this MoE tile."""
    present: set[str] = set()
    from TraceLens.ModelUtils.ast_analyze import _classify_role

    for node in _child_nodes(nodes, moe_node):
        class_name = _attr_text(node.get("attrs"), "class_name") or ""
        if not _node_is_linear(class_name):
            continue
        attr = str(node.get("id") or "").rsplit("/", 1)[-1]
        role = _attr_text(node.get("attrs"), "role") or _classify_role(attr, class_name)
        if role == "router":
            present.add("router")
        else:
            present.add("shared")
    return frozenset(present)


def _attention_variants_cover(instances: int) -> bool:
    windows = _RT.variant_windows
    if len({window for _label, _count, window in windows}) < 2:
        return False
    return instances == sum(count for _label, count, _window in windows)


def _attention_variant_rows(
    op: RooflineOp,
    node_id: str,
    section: str,
    nodes: list[Any],
    node: dict[str, Any],
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """One attention row per layer-type window, instead of one stack average."""
    rows: list[RooflineOp] = []
    opaque = not _attention_has_projection_leaves(nodes, node)
    owner_class = _attr_text(node.get("attrs"), "class_name")
    for label, count, variant_window in _RT.variant_windows:
        cost = _attention_boundary_cost(dims, bpe, window=variant_window)
        if cost is None:
            return [op]
        category, maf_type, flops, bytes_moved = cost
        variant = replace(
            op,
            name=f"{node_id}.{label}",
            section=f"{section}/{label}",
            instances=count,
        )
        rows.append(
            _finish_op(variant, category, maf_type, flops, bytes_moved, dtype, gpu_arch)
        )
        if opaque:
            rows.extend(
                _attention_extra_ops(
                    f"{section}/{label}",
                    f"{node_id}.{label}",
                    count,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                    owner_class=owner_class,
                )
            )
    return rows


def _node_is_draft_container(node: dict[str, Any]) -> bool:
    """A node whose class is the speculative layer, not one of its children."""
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    class_name = _attr_text(node.get("attrs"), "class_name") or ""
    return bool(SPECULATIVE_CLASS_RE.search(class_name))


def _draft_roots(spec: Any) -> set[str]:
    """Attribute names of the speculative modules on the top-level model."""
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    roots: set[str] = set()
    if spec is None:
        return roots
    for structure in _registry_structures(spec):
        if SPECULATIVE_CLASS_RE.search(structure.name):
            continue
        for attr, class_name in _assignment_pairs(structure):
            if SPECULATIVE_CLASS_RE.search(class_name) or SPECULATIVE_CLASS_RE.search(
                attr
            ):
                roots.add(attr)
    return roots


def _draft_graph_nodes(nodes: list[Any], spec: Any) -> list[dict[str, Any]]:
    """Nodes drawn inside the speculative module, not the backbone that shares its classes."""
    roots = _draft_roots(spec)
    if not roots:
        return []
    found: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        namespace = str(node.get("namespace") or "")
        node_id = str(node.get("id") or "")
        for root in roots:
            if (
                namespace == root
                or namespace.startswith(root + "/")
                or node_id == root
                or node_id.startswith(root + "/")
            ):
                found.append(node)
                break
    return found


def _drawn_names(nodes: list[dict[str, Any]]) -> set[str]:
    """Class, role, and id segments of nodes the graph already costed."""
    names: set[str] = set()
    for node in nodes:
        class_name = _attr_text(node.get("attrs"), "class_name") or ""
        role = _attr_text(node.get("attrs"), "role") or ""
        if class_name:
            names.add(class_name)
        if role:
            names.add(role)
        for part in re.split(r"[/:]", str(node.get("id") or "")):
            if part and not part.isdigit() and part != "seq":
                names.add(part)
    return names


def _draft_children_drawn(nodes: list[Any], spec: Any) -> bool:
    """True when the graph draws the speculative module itself."""
    return bool(_draft_graph_nodes(nodes, spec))


def _draft_child_names(spec: Any) -> set[str]:
    """Attribute and class names assigned by a speculative layer, one walk."""
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    names: set[str] = set()
    if spec is None:
        return names

    def visit(structure: Any, depth: int) -> None:
        if structure is None or depth > 4:
            return
        for attr, class_name in _assignment_pairs(structure):
            if attr in names and class_name in names:
                continue
            names.add(attr)
            names.add(class_name)
            visit(_structure_named(spec, class_name), depth + 1)

    for structure in _registry_structures(spec):
        if SPECULATIVE_CLASS_RE.search(structure.name):
            visit(structure, 0)
    return names


def _graph_contains_draft(nodes: list[Any], spec: Any = None) -> bool:
    """Whether the exported graph already draws the speculative draft stack.

    A node counts when its class matches the speculative-decoding pattern, or
    when the parsed draft class assigns that node. Assignment names come from
    the class registry, so expanded leaves count without a second stack.
    """
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    children = _draft_child_names(spec)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        texts = (
            _attr_text(node.get("attrs"), "class_name") or "",
            str(node.get("label") or ""),
            str(node.get("namespace") or ""),
            str(node.get("id") or ""),
        )
        if any(SPECULATIVE_CLASS_RE.search(text) for text in texts):
            return True
        class_name, _label, _namespace, node_id = texts
        if class_name in children or node_id.rsplit("/", 1)[-1] in children:
            return True
    return False


def _quant_side_ops(
    node_id: str,
    section: str,
    instances: int,
    input_texts: list[list[str]],
    output_text: str,
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """Activation quant and weight dequant beside a quantized linear GEMM.

    Weight-only schemes (fp4, int4) dequantize the weight tensor. An fp8 matrix
    also quantizes the activation. Modules listed in the quant config's
    exclusion list stay at the activation dtype and get neither row.
    """
    if _RT.weight_bpe is None or bpe is None or _excluded_from_quant(node_id):
        return []
    dims_only, _dtype = _split_shape_text(output_text)
    try:
        output_shape = resolve_shape(dims_only, dims)
        input_shape = resolve_shape(input_texts[0], dims) if input_texts else None
    except (DimResolutionError, IndexError):
        return []
    if not output_shape or not input_shape:
        return []
    k = input_shape[-1]
    n = output_shape[-1]
    m = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    return _quant_matrix_rows(
        node_id, section, instances, [(m, n, k)], bpe, dtype, gpu_arch
    )


def _quant_matrix_rows(
    node_id: str,
    section: str,
    instances: int,
    matrices: list[tuple[int, int, int]],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """Activation quant and weight dequant for weight matrices ``(m, n, k)``."""
    if _RT.weight_bpe is None or bpe is None or _excluded_from_quant(node_id):
        return []
    weight_elements = sum(k * n for _m, n, k in matrices)
    act_elements = sum(m * k for m, _n, k in matrices)
    if weight_elements <= 0:
        return []
    ops: list[RooflineOp] = []

    def _unary(suffix: str, elements: int, bpe_in: float, bpe_out: float) -> None:
        op = RooflineOp(
            name=f"{node_id}.{suffix}",
            section=section,
            computation=suffix,
            operation="",
            category=None,
            modeled=False,
            reason=None,
            output_elements=elements,
            instances=instances,
        )
        ops.append(
            _finish_op(
                op,
                "UnaryElementwise",
                "vector",
                UnaryElementwise.flops_func(elements),
                UnaryElementwise.bytes_func(elements, bpe_in, bpe_out),
                dtype,
                gpu_arch,
            )
        )

    _unary("weight_dequant", weight_elements, _RT.weight_bpe, bpe)
    if _RT.weight_bpe == 1:
        _unary("act_quant", act_elements, bpe, _RT.weight_bpe)
    return ops


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


def _attr_values(attrs: Any, key: str) -> tuple[str, ...]:
    """Return every value for repeatable structured graph metadata."""
    if not isinstance(attrs, list):
        return ()
    return tuple(
        str(item["value"])
        for item in attrs
        if item.get("key") == key and item.get("value")
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
    if role in ("ffn", "moe") and _RT.moe_top_k and _looks_like_moe(class_name):
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
        cost = _dispatch_op_cost(
            computation,
            input_shapes,
            output_shape,
            bpe,
            details,
            weight_bpe=_RT.weight_bpe,
        )
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
    # Weight-only quantization still executes at the activation precision.
    # A published fp4/fp8 matrix peak is used only when the weight storage is
    # that width and the selected GPU actually lists it. Otherwise the
    # activation peak remains the explicit fallback.
    weight_precision = None
    if _RT.weight_bpe is not None and _RT.weight_bpe <= 0.6:
        weight_precision = "fp4"
    elif _RT.weight_bpe == 1:
        weight_precision = "fp8"
    peaks = (gpu_arch or {}).get("max_achievable_tflops") or {}
    # SDPA and the indexer score run at the activation dtype. Only a weight
    # GEMM or grouped expert GEMM may take the fp4/fp8 matrix peak.
    if (
        category in ("GEMM", "MoE")
        and maf_type == "matrix"
        and weight_precision
        and f"matrix_{weight_precision}" in peaks
    ):
        precision = weight_precision
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
    flops = GEMM.flops_func(tokens, vocab, hidden, _RT.head_bias)
    # The head is commonly listed in modules_to_not_convert, so it stays at
    # the activation width even when the backbone weights are quantized.
    wbpe = bpe
    if _RT.weight_bpe is not None and not _excluded_from_quant("lm_head"):
        wbpe = _RT.weight_bpe
    bytes_moved = GEMM.bytes_func(
        tokens, vocab, hidden, _RT.head_bias, wbpe, wbpe, bpe, bpe
    )
    return "GEMM", "matrix", flops, bytes_moved, tokens * vocab


class _LinearWidths:
    """Constructor ``(in, out)`` sizes, by owning class and by unanimous attr.

    An attribute two classes size differently stays available on each class and
    is left out of the shared map, so one module's ``proj`` cannot size another's.
    """

    def __init__(self) -> None:
        self.by_owner: dict[tuple[str, str], tuple[int, int]] = {}
        self.agreed: dict[str, tuple[int, int]] = {}

    def get(self, attr: str, owner: str | None = None) -> tuple[int, int] | None:
        if owner:
            found = self.by_owner.get((owner, attr))
            if found is not None:
                return found
        return self.agreed.get(attr)

    def __bool__(self) -> bool:
        return bool(self.by_owner or self.agreed)


class _RuntimeIntrospection:
    """Per-run values introspected from the ArchitectureSpec (config.json +
    modeling source) so MoE / attention / quantization costs need no hardcoded
    model names. Populated by :func:`project_roofline`."""

    __slots__ = (
        "weight_bpe",
        "moe_top_k",
        "moe_intermediate",
        "moe_num_experts",
        "num_shared_experts",
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
        "causal_attention",
        "qk_norm",
        "uses_rope",
        "kv_lora_rank",
        "q_lora_rank",
        "index_topk",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_n_heads",
        "index_head_dim",
        "attention_bias",
        "mlp_bias",
        "head_bias",
        "linear_widths",
        "linear_owners",
        "variant_windows",
        "engine_miss",
        "module_dims",
        "quant_excluded",
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
        self.num_shared_experts = None
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
        self.causal_attention = True
        self.qk_norm = False
        self.uses_rope = False
        self.kv_lora_rank = None
        self.q_lora_rank = None
        self.index_topk = None
        self.qk_nope_head_dim = None
        self.qk_rope_head_dim = None
        self.v_head_dim = None
        self.index_n_heads = None
        self.index_head_dim = None
        self.attention_bias = False
        self.mlp_bias = False
        self.head_bias = False
        self.linear_widths = _LinearWidths()
        self.linear_owners = {}
        self.variant_windows = ()
        self.engine_miss = None
        self.module_dims = None
        self.quant_excluded = ()
        self.sconv_kernel = None
        self.dense_intermediate = None
        self.mtp_layers = None


_RT = _RuntimeIntrospection()


def _excluded_from_quant(name: str) -> bool:
    """Whether *name* is kept at the activation dtype by the quant config."""
    lowered = name.lower()
    return any(part and part in lowered for part in _RT.quant_excluded)


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
    quant = cfg.get("quantization_config")
    _RT.weight_bpe = _introspect_weight_bpe(quant)
    excluded = ()
    if isinstance(quant, dict):
        raw = (
            quant.get("modules_to_not_convert")
            or quant.get("ignore")
            or quant.get("exclude_layers")
            or ()
        )
        if isinstance(raw, str):
            raw = (raw,)
        excluded = tuple(str(item).lower() for item in raw)
    _RT.quant_excluded = excluded
    _RT.moe_top_k = spec.num_experts_per_tok
    _RT.moe_intermediate = spec.moe_intermediate_size or spec.intermediate_size
    _RT.moe_num_experts = spec.num_experts
    _RT.num_shared_experts = spec.num_shared_experts
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
    _RT.qk_norm = bool(tcfg.get("qk_norm") or tcfg.get("use_qk_norm"))
    _RT.uses_rope = getattr(spec, "positional_encoding", None) == "RoPE"
    _RT.kv_lora_rank = _config_int(tcfg, "kv_lora_rank")
    _RT.q_lora_rank = _config_int(tcfg, "q_lora_rank")
    _RT.index_topk = _config_int(tcfg, "index_topk")
    _RT.qk_nope_head_dim = _config_int(tcfg, "qk_nope_head_dim")
    _RT.qk_rope_head_dim = _config_int(tcfg, "qk_rope_head_dim")
    _RT.v_head_dim = _config_int(tcfg, "v_head_dim")
    _RT.index_n_heads = _config_int(tcfg, "index_n_heads", "index_head_number")
    _RT.index_head_dim = _config_int(tcfg, "index_head_dim")
    _RT.attention_bias = tcfg.get("attention_bias") is True
    _RT.mlp_bias = tcfg.get("mlp_bias") is True
    _RT.head_bias = tcfg.get("lm_head_bias", tcfg.get("head_bias")) is True
    _RT.variant_windows = _variant_windows(
        tcfg, _RT.sliding_window
    ) or _windows_from_variants(spec, tcfg, _RT.sliding_window)
    _apply_source_widths(spec)
    # Decoder-only checkpoints are causal by construction.  An explicit
    # ``is_causal``/``causal`` config value wins, while encoder and
    # encoder-decoder configs are non-causal unless they say otherwise.
    causal = tcfg.get("is_causal", tcfg.get("causal"))
    if isinstance(causal, bool):
        _RT.causal_attention = causal
    else:
        architectures = " ".join(getattr(spec, "architectures", None) or ())
        model_type = str(getattr(spec, "model_type", "")).lower()
        _RT.causal_attention = not any(
            marker in f"{architectures} {model_type}".lower()
            for marker in ("encoder", "bert", "t5", "vision")
        )
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


# Constructor scalars that override the checkpoint when the modeling source
# resolved them to a concrete int. Intermediate widths are taken only from a
# class whose name says which MLP they belong to.
_SOURCE_SCALAR_FIELDS = {
    "head_dim": "attn_head_dim",
    "attention_head_dim": "attn_head_dim",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "v_head_dim": "v_head_dim",
    "kv_lora_rank": "kv_lora_rank",
    "q_lora_rank": "q_lora_rank",
    "index_topk": "index_topk",
    "index_head_dim": "index_head_dim",
    "index_n_heads": "index_n_heads",
    "index_head_number": "index_n_heads",
    "num_attention_heads": "attn_heads",
    "num_heads": "attn_heads",
    "num_key_value_heads": "attn_kv_heads",
    "moe_intermediate_size": "moe_intermediate",
    "intermediate_size": "dense_intermediate",
}


def _local_layer_indices(config: dict[str, Any], n_layers: int) -> set[int]:
    """Layer indices the checkpoint marks as windowed, when it lists them."""
    raw = config.get("local_layer_ids")
    if not isinstance(raw, list):
        return set()
    return {index for index in raw if isinstance(index, int) and 0 <= index < n_layers}


def _variant_windows(
    config: dict[str, Any], sliding: int | None
) -> tuple[tuple[str, int, int | None], ...]:
    """Per-type attention windows from ``layer_types`` and ``local_layer_ids``.

    A listed local index uses the checkpoint window. Every other index attends
    the full context. One window for the whole stack returns nothing, because
    ``sliding_window`` already describes that case.
    """
    types = config.get("layer_types")
    if not isinstance(types, list) or not types or not sliding:
        return ()
    local = _local_layer_indices(config, len(types))
    if not local:
        return ()
    groups: list[tuple[str, int, int | None]] = []
    for index, item in enumerate(types):
        label = str(item)
        window = sliding if index in local else None
        if groups and groups[-1][0] == label and groups[-1][2] == window:
            groups[-1] = (label, groups[-1][1] + 1, window)
        else:
            groups.append((label, 1, window))
    if len({window for _label, _count, window in groups}) < 2:
        return ()
    return tuple(groups)


def _windows_from_variants(
    spec: Any, config: dict[str, Any], sliding: int | None
) -> tuple[tuple[str, int, int | None], ...]:
    """The same split from layer-variant indices when ``layer_types`` is absent."""
    if not sliding:
        return ()
    variants = list(getattr(spec, "layer_variants", None) or [])
    indices = [
        index
        for variant in variants
        for index in (getattr(variant, "layer_indices", None) or [])
        if isinstance(index, int)
    ]
    if not indices:
        return ()
    local = _local_layer_indices(config, max(indices) + 1)
    if not local:
        return ()
    groups: list[tuple[str, int, int | None]] = []
    for variant in variants:
        layer_indices = [
            index
            for index in (getattr(variant, "layer_indices", None) or [])
            if isinstance(index, int)
        ]
        count = getattr(variant, "count", None) or len(layer_indices)
        if not count or not layer_indices:
            continue
        label = str(getattr(variant, "label", "") or "layer")
        window = sliding if any(index in local for index in layer_indices) else None
        groups.append((label, int(count), window))
    if len({window for _label, _count, window in groups}) < 2:
        return ()
    return tuple(groups)


def _overlay_source_scalars(registry: Any, class_names: set[str] | None = None) -> dict:
    """Copy concrete constructor ints onto ``_RT``. Returns the fields written.

    A name is applied only when every included class agrees on its value.
    Disagreement leaves the checkpoint value in place.
    """
    found: dict[str, dict[int, list[str]]] = {}
    scalars = getattr(registry, "scalar_by_class", None) or {}
    for class_name, values in scalars.items():
        if class_names is not None and class_name not in class_names:
            continue
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            field_name = _SOURCE_SCALAR_FIELDS.get(str(key))
            if field_name is None or isinstance(value, bool):
                continue
            if not isinstance(value, int) or value <= 0:
                continue
            found.setdefault(field_name, {}).setdefault(value, []).append(
                str(class_name)
            )
    written: dict[str, int] = {}
    for field_name, by_value in found.items():
        if len(by_value) != 1:
            continue
        value = next(iter(by_value))
        setattr(_RT, field_name, value)
        written[field_name] = value
    return written


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _collect_linear_widths(registry: Any) -> _LinearWidths:
    """Record every resolved linear, and the attrs whose sizes all agree."""
    widths = _LinearWidths()
    found: dict[str, set[tuple[int, int]]] = {}
    for key, spec in (getattr(registry, "linear", None) or {}).items():
        class_name, attr = key if isinstance(key, tuple) and len(key) == 2 else ("", "")
        if not attr:
            continue
        inn = _positive_int(getattr(spec, "in_features", None))
        out = _positive_int(getattr(spec, "out_features", None))
        if inn is None or out is None:
            continue
        widths.by_owner[(str(class_name), str(attr))] = (inn, out)
        found.setdefault(str(attr), set()).add((inn, out))
    widths.agreed = {
        attr: next(iter(pairs)) for attr, pairs in found.items() if len(pairs) == 1
    }
    return widths


def _linear_parallel_style(class_name: str) -> str | None:
    """The engine shard axis for a linear class, from the shared style table."""
    from TraceLens.ModelUtils.parallel_plan import _ENGINE_LINEAR_STYLES

    if not class_name:
        return None
    style = _ENGINE_LINEAR_STYLES.get(class_name)
    if style:
        return style
    for name in sorted(_ENGINE_LINEAR_STYLES, key=len, reverse=True):
        if name in class_name:
            return _ENGINE_LINEAR_STYLES[name]
    return None


def _adopt_linear_width(
    global_in: int,
    global_out: int,
    formula_k: int,
    formula_n: int,
    linear_class: str,
) -> tuple[int, int]:
    """Per-GPU ``(k, n)`` from a constructor width.

    An engine parallel linear shards the axis its style names. A plain linear
    keeps a formula axis that is already smaller than the constructor size and
    takes the constructor size on every other axis.
    """
    style = _linear_parallel_style(linear_class)
    if style == "rowwise":
        return _shard(global_in, _RT.tp), global_out
    if style in ("colwise", "packed_colwise", "embedding_rowwise"):
        return global_in, _shard(global_out, _RT.tp)
    inn = (
        _shard(global_in, _RT.tp) if _RT.tp > 1 and formula_k < global_in else global_in
    )
    out = (
        _shard(global_out, _RT.tp)
        if _RT.tp > 1 and formula_n < global_out
        else global_out
    )
    return inn, out


def _index_linear_owners(spec: Any) -> dict[str, tuple[tuple[str, str], ...]]:
    """``owner class -> ((attr, linear class), ...)`` for its linear assignments."""
    from TraceLens.ModelUtils.ast_analyze import displays_as_linear

    owners: dict[str, tuple[tuple[str, str], ...]] = {}
    for structure in _registry_structures(spec):
        rows = tuple(
            (attr, class_name)
            for attr, class_name in _assignment_pairs(structure)
            if displays_as_linear(attr, class_name)
        )
        if rows:
            owners[structure.name] = rows
    return owners


def _apply_source_widths(spec: Any) -> None:
    """Prefer constructor-resolved widths, and record a missing engine file.

    A vLLM/SGLang run with no parsed modeling source used to emit the same
    config rows as a successful parse. The projection note says which one ran.
    """
    _RT.engine_miss = None
    _RT.module_dims = None
    engine = getattr(spec, "engine", None)
    class_registry = getattr(spec, "class_registry", None) or {}
    if engine and not class_registry:
        _RT.engine_miss = (
            f"{engine} modeling source was not loaded; "
            "roofline widths come from the checkpoint config"
        )
        return
    if not class_registry:
        return
    try:
        from TraceLens.ModelUtils.shape_inference import ModuleDimRegistry

        context = ShapeContext.from_spec(spec)
        registry = ModuleDimRegistry.from_registry(
            class_registry,
            config=getattr(spec, "raw_config", None) or {},
            context=context,
        )
    except Exception:
        _RT.engine_miss = (
            "modeling source could not be resolved into constructor widths; "
            "roofline widths come from the checkpoint config"
        )
        return
    _RT.module_dims = registry
    written = _overlay_source_scalars(registry)
    _RT.linear_widths = _collect_linear_widths(registry)
    _RT.linear_owners = _index_linear_owners(spec)
    if (
        engine
        and not written
        and not _RT.linear_widths
        and not registry.known_classes()
    ):
        _RT.engine_miss = (
            f"{engine} modeling source had no usable constructor widths; "
            "roofline widths come from the checkpoint config"
        )


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
    text = (class_name or "").lower()
    return "moe" in text or "expert" in text


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
    if _is_embedding_lookup(computation):
        return True
    return _is_short_conv(computation, "") or bool(
        re.search(r"(?:causal_)?conv(?:transpose)?[123]d", computation, re.I)
    )


def _shard(n: int, degree: int) -> int:
    """One GPU's floor share of *n* under a *degree*-way split (never below 1).

    Expert-parallel token counts use the floor: a rank cannot route a fraction
    of a token. Head counts use :func:`_shard_heads`, which keeps the remainder
    on the busiest rank.
    """
    return max(1, n // degree) if degree and degree > 1 else n


def _shard_heads(n: int, degree: int) -> int:
    """Head count on the busiest GPU.

    Ranks that receive a remainder hold one more head than the floor. A count
    smaller than the degree is still one head on the ranks that have one.
    """
    if not degree or degree <= 1 or n <= 0:
        return n
    if n <= degree:
        return 1
    return (n + degree - 1) // degree


_WINDOW_UNSET = object()


def _attention_boundary_cost(dims, bpe, window=_WINDOW_UNSET):
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
    h_q = _shard_heads(_RT.attn_heads, _RT.tp)
    h_kv = _shard_heads(_RT.attn_kv_heads or _RT.attn_heads, _RT.tp)
    d_h = _RT.attn_head_dim
    d_qk = d_h
    d_v = _RT.v_head_dim or d_h
    # Latent attention attends one compressed KV vector, not a sharded head
    # stack of the full head dim. nope+rope is the query width; without those
    # fields the latent rank itself is the score dimension.
    if _RT.kv_lora_rank:
        h_kv = 1
        rope = _RT.qk_rope_head_dim or 0
        # Absorbed latent attention scores against the compressed key
        # (kv_lora_rank + rope), which is not the query's nope+rope width.
        d_qk = _RT.kv_lora_rank + rope
        d_v = _RT.v_head_dim or _RT.kv_lora_rank
    if window is not _WINDOW_UNSET:
        eff_n_kv = min(ctx, window) if isinstance(window, int) and window > 0 else ctx
    elif _RT.variant_windows:
        total = sum(count for _label, count, _window in _RT.variant_windows) or 1
        weighted = 0.0
        for _label, count, variant_window in _RT.variant_windows:
            length = (
                min(ctx, variant_window)
                if isinstance(variant_window, int) and variant_window > 0
                else ctx
            )
            weighted += count * length
        eff_n_kv = weighted / total
    else:
        n_layers = _RT.n_layers or 1
        if _RT.n_local_layers is not None:
            n_local = _RT.n_local_layers
        else:
            n_local = n_layers if _RT.sliding_window else 0
        n_global = max(n_layers - n_local, 0)
        # index_topk is the indexed-attention length. A per-layer window already
        # says how far that layer attends, so the top-k does not replace it.
        if _RT.index_topk and not _RT.sliding_window and n_local == 0:
            ctx = min(ctx, _RT.index_topk)
        kv_local = min(ctx, _RT.sliding_window) if _RT.sliding_window else ctx
        eff_n_kv = (n_local * kv_local + n_global * ctx) / n_layers if n_layers else ctx
    # ``SDPA`` only accepts causal masking when query and key/value lengths
    # agree.  During warm-cache decode Q is a suffix of KV and the engine
    # applies the model's causal policy in its cache kernel; its FLOP count is
    # unchanged by the mask.  Cold/prefill self-attention therefore takes the
    # model-declared causal path instead of the historical forced ``False``.
    causal = _RT.causal_attention and past == 0 and s == eff_n_kv
    flops = SDPA.flops_func(b, s, h_q, eff_n_kv, h_kv, d_qk, d_v, causal)
    bytes_moved = (
        SDPA.bytes_func(b, s, h_q, eff_n_kv, h_kv, d_qk, d_v, causal, bpe)
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
    owner_class: str | None = None,
    skip: set[str] | None = None,
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
    h_q = _shard_heads(_RT.attn_heads, _RT.tp)
    h_kv = _shard_heads(_RT.attn_kv_heads or _RT.attn_heads, _RT.tp)
    d_h = _RT.attn_head_dim
    nope = _RT.qk_nope_head_dim or 0
    rope = _RT.qk_rope_head_dim or 0
    v_head = _RT.v_head_dim or 0
    # Latent attention's query and value widths are the declared nope/rope/v
    # ranks. head_dim remains the width when those ranks are absent.
    q_head = (nope + rope) if (nope or rope) else d_h
    v_width = v_head or q_head
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    q_out = h_q * q_head
    kv_out = h_kv * d_h
    o_in = h_q * v_width
    ops: list[RooflineOp] = []

    def _gemm_row(
        suffix: str,
        n: int,
        k: int,
        out_elems: int,
        owner: str | None = None,
        linear_class: str = "",
    ) -> list[RooflineOp]:
        width = _RT.linear_widths.get(suffix, owner)
        if width:
            k, n = _adopt_linear_width(width[0], width[1], k, n, linear_class)
            out_elems = tokens * n
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
        flops = GEMM.flops_func(tokens, n, k, _RT.attention_bias)
        byt = (
            GEMM.bytes_func(tokens, n, k, _RT.attention_bias, wbpe, wbpe, bpe, bpe)
            if bpe is not None
            else None
        )
        finished = _finish_op(op, "GEMM", "matrix", flops, byt, dtype, gpu_arch)
        finished.cost_source = "source" if width else "config"
        rows = [
            finished,
            *_quant_matrix_rows(
                f"{node_id}.{suffix}",
                section,
                instances,
                [(tokens, n, k)],
                bpe,
                dtype,
                gpu_arch,
            ),
        ]
        for row in rows[1:]:
            row.cost_source = finished.cost_source
        return rows

    # The attention class's own linear assignments are the projections. The
    # packed or latent template is the fallback when that class declares none.
    owned = _RT.linear_owners.get(owner_class or "", ())
    owned_widths = [
        (attr, linear_class, _RT.linear_widths.get(attr, owner_class))
        for attr, linear_class in owned
        if not skip or attr not in skip
    ]
    use_owner = bool(owned) and all(
        width is not None for _attr, _cls, width in owned_widths
    )
    if use_owner:
        for attr, linear_class, width in owned_widths:
            if width is None:
                continue
            inn, out = width
            ops.extend(
                _gemm_row(attr, out, inn, tokens * out, owner_class, linear_class)
            )
    elif _RT.kv_lora_rank:
        # Latent attention declares kv_lora_rank. Cost the low-rank Q/KV
        # projections from those config widths. Otherwise vLLM/SGLang pack Q, K,
        # and V into one column-parallel projection.
        kv_b_out = (
            h_q * ((nope or d_h) + (v_head or d_h)) if (nope or v_head) else kv_out
        )
        kv_a_out = _RT.kv_lora_rank + rope
        if _RT.q_lora_rank:
            if not skip or "q_a_proj" not in skip:
                ops.extend(
                    _gemm_row(
                        "q_a_proj", _RT.q_lora_rank, hidden, tokens * _RT.q_lora_rank
                    )
                )
            if not skip or "q_b_proj" not in skip:
                ops.extend(
                    _gemm_row("q_b_proj", q_out, _RT.q_lora_rank, tokens * q_out)
                )
        elif not skip or "q_proj" not in skip:
            ops.extend(_gemm_row("q_proj", q_out, hidden, tokens * q_out))
        if not skip or "kv_a_proj" not in skip:
            ops.extend(_gemm_row("kv_a_proj", kv_a_out, hidden, tokens * kv_a_out))
        if not skip or "kv_b_proj" not in skip:
            ops.extend(
                _gemm_row("kv_b_proj", kv_b_out, _RT.kv_lora_rank, tokens * kv_b_out)
            )
    else:
        if not skip or "qkv_proj" not in skip:
            ops.extend(
                _gemm_row(
                    "qkv_proj",
                    q_out + 2 * kv_out,
                    hidden,
                    tokens * (q_out + 2 * kv_out),
                )
            )
    if not use_owner and (not skip or "o_proj" not in skip):
        ops.extend(_gemm_row("o_proj", hidden, o_in, tokens * hidden))
    # These operations are normally hidden by a fused engine attention
    # boundary.  Publish them only on that boundary (expanded graph leaves are
    # handled by the regular dispatch path).
    if _RT.qk_norm:
        for suffix, elements in (
            ("q_norm", tokens * q_out),
            ("k_norm", tokens * kv_out),
        ):
            ops.append(
                _analytical_op(
                    f"{node_id}.{suffix}",
                    section,
                    suffix,
                    instances,
                    "Normalization (RMSNorm)",
                    "vector",
                    _rmsnorm_flops(elements, d_h, True),
                    (
                        _rmsnorm_bytes(elements, d_h, True, bpe, bpe)
                        if bpe is not None
                        else None
                    ),
                    elements,
                    dtype,
                    gpu_arch,
                )
            )
    if _RT.uses_rope:
        rope_elems = tokens * (q_out + kv_out)
        rope_bytes = 2 * rope_elems * bpe if bpe is not None else None
        ops.append(
            _analytical_op(
                f"{node_id}.rope",
                section,
                "rope",
                instances,
                "RoPE",
                "vector",
                3.0 * rope_elems,
                rope_bytes,
                rope_elems,
                dtype,
                gpu_arch,
            )
        )

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
        ops.extend(_gemm_row("rel_r_proj", r_out, hidden, tokens * r_out))
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
    if _RT.index_n_heads and _RT.index_head_dim:
        h_i = _shard_heads(_RT.index_n_heads, _RT.tp)
        d_i = _RT.index_head_dim
        ops.extend(_gemm_row("indexer_q", h_i * d_i, hidden, tokens * h_i * d_i))
        ops.extend(_gemm_row("indexer_k", d_i, hidden, tokens * d_i))
        try:
            score_ctx = _resolve_past(dims) + resolve_dim(dims["S"], dims)
        except (KeyError, DimResolutionError):
            score_ctx = 0
        if score_ctx:
            # The score is the query-key product. Top-k then reads those scores
            # and writes the selected indices.
            score_m = tokens * h_i
            score_flops = GEMM.flops_func(score_m, score_ctx, d_i, bias=False)
            score_bytes = (
                GEMM.bytes_func(score_m, score_ctx, d_i, False, bpe, bpe, bpe, bpe)
                if bpe is not None
                else None
            )
            ops.append(
                _analytical_op(
                    f"{node_id}.indexer_score",
                    section,
                    "indexer_score",
                    instances,
                    "GEMM",
                    "matrix",
                    score_flops,
                    score_bytes,
                    score_m * score_ctx,
                    dtype,
                    gpu_arch,
                )
            )
            topk = _RT.index_topk or 1
            score_elems = score_m * score_ctx
            index_elems = score_m * topk
            ops.append(
                _analytical_op(
                    f"{node_id}.indexer_topk",
                    section,
                    "indexer_topk",
                    instances,
                    "Reduce",
                    "vector",
                    float(score_elems),
                    ((score_elems + index_elems) * bpe if bpe is not None else None),
                    index_elems,
                    dtype,
                    gpu_arch,
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
    gate_up_flops = GEMM.flops_func(tokens, 2 * intermediate, hidden, _RT.mlp_bias)
    gate_up_bytes = GEMM.bytes_func(
        tokens, 2 * intermediate, hidden, _RT.mlp_bias, wbpe, wbpe, bpe, bpe
    )
    down_flops = GEMM.flops_func(tokens, hidden, intermediate, _RT.mlp_bias)
    down_bytes = GEMM.bytes_func(
        tokens, hidden, intermediate, _RT.mlp_bias, wbpe, wbpe, bpe, bpe
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
    finished = _finish_op(op, category, maf_type, flops, bytes_moved, dtype, gpu_arch)
    finished.cost_source = "config"
    return finished


@dataclass(frozen=True)
class _DraftLayer:
    """The speculative block read from modeling source, not the fallback template."""

    name: str
    norms: tuple[str, ...]
    projections: tuple[tuple[str, str], ...]
    has_attention: bool
    ffn_role: str | None
    convs: tuple[str, ...]
    classes: tuple[str, ...] = ()
    has_head: bool = False
    attention_class: str | None = None
    ffn_linears: tuple[str, ...] = ()


def _registry_structures(spec: Any):
    registry = getattr(spec, "class_registry", None) or {}
    for structure in registry.values():
        name = getattr(structure, "name", None)
        assignments = getattr(structure, "init_assignments", None)
        if isinstance(name, str) and isinstance(assignments, dict):
            yield structure


def _structure_named(spec: Any, class_name: str | None):
    if not class_name:
        return None
    registry = getattr(spec, "class_registry", None) or {}
    structure = registry.get(class_name)
    assignments = getattr(structure, "init_assignments", None)
    if isinstance(getattr(structure, "name", None), str) and isinstance(
        assignments, dict
    ):
        return structure
    return None


def _assignment_pairs(structure: Any):
    for attr, class_name in (
        getattr(structure, "init_assignments", None) or {}
    ).items():
        base = str(attr).split("@", 1)[0]
        if base.startswith("@"):
            continue
        yield base, str(class_name or "")


def _draft_ffn_role(attr: str, class_name: str) -> str | None:
    """Dense or MoE for one draft assignment.

    ``self.mlp = FusedMoE(...)`` keeps the attribute name of a dense MLP, so the
    class name decides. ``_looks_like_moe`` catches the engine wrappers
    (``FusedMoE``) that the strict MoE-class pattern does not.
    """
    from TraceLens.ModelUtils.ast_analyze import ffn_role_for_class

    if _looks_like_moe(class_name):
        return "moe"
    role = ffn_role_for_class(attr, class_name)
    return role if role in ("ffn", "moe") else None


def _variant_ffn_role(spec: Any) -> str | None:
    """The feed-forward the backbone uses on its last layer.

    A draft that reuses the backbone decoder continues that tail: a dense prefix
    followed by MoE does not make the draft dense.
    """
    variants = getattr(spec, "layer_variants", None) or []
    if not variants:
        return None

    def _tail(variant: Any) -> int:
        indices = getattr(variant, "layer_indices", None) or []
        return max(indices) if indices else 0

    last = max(variants, key=_tail)
    return _draft_ffn_role(
        getattr(last, "ffn_attr", None) or "mlp",
        getattr(last, "ffn_class", None) or "",
    )


def _ffn_roles_of(structure: Any, spec: Any) -> set[str]:
    roles: set[str] = set()
    options = getattr(structure, "init_assignment_options", None) or {}
    for attr, class_name in _assignment_pairs(structure):
        names = options.get(attr) or [class_name]
        found = {
            role
            for name in names
            if (role := _draft_ffn_role(attr, str(name or ""))) is not None
        }
        if len(found) > 1:
            variant = _variant_ffn_role(spec)
            if variant:
                roles.add(variant)
            continue
        roles.update(found)
    return roles


def _is_draft_structure(structure: Any) -> bool:
    from TraceLens.ModelUtils.ast_analyze import SPECULATIVE_CLASS_RE

    return bool(SPECULATIVE_CLASS_RE.search(structure.name))


def _block_children(spec: Any, structure: Any) -> list[Any]:
    """Decoder-like submodules whose own assignments are the draft's compute."""
    from TraceLens.ModelUtils.ast_analyze import _classify_role

    children = []
    for _attr, class_name in _assignment_pairs(structure):
        child = _structure_named(spec, class_name)
        if child is None or child is structure:
            continue
        pairs = list(_assignment_pairs(child))
        if any(
            _classify_role(attr, name) == "attention" or _draft_ffn_role(attr, name)
            for attr, name in pairs
        ):
            children.append(child)
    return children


def _draft_class_names(spec: Any, roots: list[Any]) -> tuple[str, ...]:
    """The draft layer and every class it assigns, so nested widths are visible."""
    names: list[str] = []

    def visit(structure: Any, depth: int) -> None:
        if structure is None or depth > 4 or structure.name in names:
            return
        names.append(structure.name)
        for _attr, class_name in _assignment_pairs(structure):
            visit(_structure_named(spec, class_name), depth + 1)

    for root in roots:
        visit(root, 0)
    return tuple(names)


def _resolve_draft_layer(spec: Any) -> _DraftLayer | None:
    """The speculative layer declared by the modeling source, if it has one."""
    from TraceLens.ModelUtils.ast_analyze import _classify_role, displays_as_linear

    if spec is None:
        return None
    candidates = [
        item for item in _registry_structures(spec) if _is_draft_structure(item)
    ]
    if not candidates:
        return None

    def _score(structure: Any) -> tuple[int, int]:
        children = _block_children(spec, structure)
        attention = any(
            _classify_role(attr, name) == "attention"
            for owner in (structure, *children)
            for attr, name in _assignment_pairs(owner)
        )
        return (
            int(attention or bool(children)),
            len(list(_assignment_pairs(structure))),
        )

    layer = max(candidates, key=_score)
    for child in _block_children(spec, layer):
        if _is_draft_structure(child) and _score(child) > _score(layer):
            layer = child
    owners = [layer, *_block_children(spec, layer)]
    norms: list[str] = []
    projections: list[tuple[str, str]] = []
    convs: list[str] = []
    has_attention = False
    has_head = False
    attention_class: str | None = None
    ffn_linears: list[str] = []
    for owner in owners:
        for attr, class_name in _assignment_pairs(owner):
            role = _classify_role(attr, class_name)
            if role == "norm" and attr not in norms:
                norms.append(attr)
            elif role == "attention":
                has_attention = True
                attention_class = attention_class or class_name
            elif role == "head":
                has_head = True
            elif displays_as_linear(attr, class_name) and attr not in {
                name for name, _owner in projections
            }:
                projections.append((attr, owner.name))
            elif "Conv" in class_name and attr not in convs:
                convs.append(attr)
            if role in ("ffn", "moe"):
                child = _structure_named(spec, class_name)
                for child_attr, child_class in (
                    _assignment_pairs(child) if child else []
                ):
                    if (
                        displays_as_linear(child_attr, child_class)
                        and _classify_role(child_attr, child_class) != "router"
                        and child_attr not in ffn_linears
                    ):
                        ffn_linears.append(child_attr)
    roles = set()
    for owner in owners:
        roles.update(_ffn_roles_of(owner, spec))
    if "moe" in roles:
        ffn_role: str | None = "moe"
    elif "ffn" in roles:
        ffn_role = "ffn"
    else:
        ffn_role = None
    if not (norms or projections) or not (has_attention or ffn_role):
        return None
    return _DraftLayer(
        name=layer.name,
        norms=tuple(norms),
        projections=tuple(projections),
        has_attention=has_attention,
        ffn_role=ffn_role,
        convs=tuple(convs),
        classes=_draft_class_names(spec, owners),
        has_head=has_head,
        attention_class=attention_class,
        ffn_linears=tuple(ffn_linears),
    )


def _push_draft_widths(draft: _DraftLayer) -> dict[str, Any]:
    """Apply the draft class's constructor ints over the backbone widths."""
    saved: dict[str, Any] = {}
    scalars = getattr(_RT.module_dims, "scalar_by_class", None) or {}
    for class_name in draft.classes:
        for key, value in (scalars.get(class_name) or {}).items():
            field_name = _SOURCE_SCALAR_FIELDS.get(str(key))
            if (
                field_name is None
                or field_name in saved
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                continue
            current = getattr(_RT, field_name)
            if current == value:
                continue
            saved[field_name] = current
            setattr(_RT, field_name, value)
    return saved


def _tag_cost_source(ops: list[RooflineOp], cost_source: str) -> list[RooflineOp]:
    for op in ops:
        op.cost_source = cost_source
    return ops


_ATTENTION_WIDTH_FIELDS = frozenset(
    {
        "attn_heads",
        "attn_kv_heads",
        "attn_head_dim",
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_topk",
        "index_head_dim",
        "index_n_heads",
    }
)


def _overridden(fields: frozenset[str], saved: dict[str, Any]) -> str:
    return "source" if fields & saved.keys() else "config"


def _draft_depth_ops(
    draft: _DraftLayer,
    section: str,
    node_id: str,
    dims: dict[str, Any],
    tokens: int,
    hidden: int,
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
    saved: dict[str, Any] | None = None,
    drawn: list[dict[str, Any]] | None = None,
) -> list[RooflineOp]:
    """One speculative depth whose ops are the parsed draft layer's own modules.

    ``drawn`` is the graph's nodes inside the draft. Those leaves are not
    emitted again. Attention and expert GEMMs the graph inlined away still are.
    """
    ops: list[RooflineOp] = []
    drawn_names = _drawn_names(drawn) if drawn else set()
    partial = drawn is not None
    norm_elems = tokens * hidden
    norm_flops = _rmsnorm_flops(norm_elems, hidden, True)
    norm_bytes = (
        _rmsnorm_bytes(norm_elems, hidden, True, bpe, bpe) if bpe is not None else None
    )
    for name in draft.norms if not partial else ():
        ops.append(
            _analytical_op(
                f"{node_id}.{name}",
                section,
                name,
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
    overridden = saved or {}
    for name, owner in draft.projections if not partial else ():
        width = _RT.linear_widths.get(name, owner)
        if width and width != (hidden, hidden):
            k, n = width
            origin = "source"
        else:
            k, n = hidden, hidden
            origin = "config"
        proj_flops = GEMM.flops_func(tokens, n, k, bias=False)
        proj_bytes = (
            GEMM.bytes_func(tokens, n, k, False, bpe, bpe, bpe, bpe)
            if bpe is not None
            else None
        )
        proj = _analytical_op(
            f"{node_id}.{name}",
            section,
            name,
            1,
            "GEMM",
            "matrix",
            proj_flops,
            proj_bytes,
            tokens * n,
            dtype,
            gpu_arch,
        )
        proj.cost_source = origin
        ops.append(proj)
        for row in _quant_matrix_rows(
            f"{node_id}.{name}", section, 1, [(tokens, n, k)], bpe, dtype, gpu_arch
        ):
            row.cost_source = origin
            ops.append(row)
    attention_drawn = partial and (
        "attention" in drawn_names or (draft.attention_class or "") in drawn_names
    )
    if draft.has_attention and not attention_drawn:
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
                    section,
                    f"{node_id}.attn",
                    1,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                    owner_class=draft.attention_class,
                    skip=drawn_names if partial else None,
                )
            )
            attn_source = _overridden(_ATTENTION_WIDTH_FIELDS, overridden)
            for row in ops:
                if row.cost_source == "source":
                    continue
                if row.computation == "attn" or row.name.startswith(f"{node_id}.attn"):
                    row.cost_source = attn_source
    ffn_drawn = partial and (
        (draft.ffn_role or "") in drawn_names
        or bool(draft.ffn_linears)
        and all(name in drawn_names for name in draft.ffn_linears)
    )
    if draft.ffn_role == "moe" and not ffn_drawn:
        moe = _moe_cost(dims, _RT.moe_top_k, _RT.moe_intermediate, bpe)
        if moe is not None:
            category, maf_type, flops, bytes_moved, out_elems = moe
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
            ops.extend(
                _moe_extra_ops(
                    section,
                    f"{node_id}.mlp",
                    1,
                    dims,
                    bpe,
                    dtype,
                    gpu_arch,
                    present=frozenset(
                        name for name in ("router", "shared") if name in drawn_names
                    ),
                )
            )
            ffn_source = _overridden(frozenset({"moe_intermediate"}), overridden)
            for row in ops:
                if row.name.startswith(f"{node_id}.mlp"):
                    row.cost_source = ffn_source
    elif draft.ffn_role == "ffn" and not ffn_drawn:
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
            for row in ops:
                if row.name.startswith(f"{node_id}.mlp"):
                    row.cost_source = _overridden(
                        frozenset({"dense_intermediate"}), overridden
                    )
    if _RT.sconv_kernel:
        channels = _shard(hidden, _RT.tp)
        for name in draft.convs:
            ops.append(
                _short_conv_op(
                    f"{node_id}.{name}",
                    section,
                    name,
                    1,
                    tokens,
                    channels,
                    bpe,
                    dtype,
                    gpu_arch,
                )
            )
    if draft.has_head:
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


def _mtp_stack_ops(
    dims: dict[str, Any],
    num_predicted_tokens: int,
    gpu_arch: dict[str, Any] | None,
    activation: tuple[float | None, str | None],
    spec: Any = None,
    drawn: list[dict[str, Any]] | None = None,
) -> list[RooflineOp]:
    """One draft depth per predicted token, as its own stack.

    Capped at the depth the checkpoint declares (``mtp_config`` /
    ``num_mtp_layers``). When the modeling source declares a draft layer, each
    depth costs that layer: its norms, its linear projections, its attention,
    and its own dense or MoE feed-forward. A row is ``cost_source="source"``
    when the draft class's constructor overrides that width, and ``"config"``
    when the width is the backbone checkpoint. With no draft layer in the
    source, the depth is the explicit fallback — dual RMSNorm, a 2H→H
    projection, one attention, a dense MLP, and the LM head — marked
    ``cost_source="fallback"``. The parsed layer includes an LM head only when
    it assigns one.
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
    draft = _resolve_draft_layer(spec)
    ops: list[RooflineOp] = []
    for index in range(count):
        section = f"mtp/{index}"
        node_id = f"mtp/{index}"
        if draft is not None:
            saved = _push_draft_widths(draft)
            try:
                ops.extend(
                    _draft_depth_ops(
                        draft,
                        section,
                        node_id,
                        dims,
                        tokens,
                        hidden,
                        bpe,
                        dtype,
                        gpu_arch,
                        saved,
                        drawn,
                    )
                )
            finally:
                for field_name, value in saved.items():
                    setattr(_RT, field_name, value)
            continue
        # No draft layer in the source. This template is the explicit fallback:
        # two RMSNorms, a 2H→H projection, attention, and a dense MLP.
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
    if draft is None:
        _tag_cost_source(ops, "fallback")
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
    experts_here = _shard(_RT.moe_num_experts, _RT.ep) if _RT.moe_num_experts else 0
    groups = min(experts_here, active_tokens) if experts_here else 1
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    # gate_up_proj: [active_tokens, hidden] x per-expert [hidden, 2*intermediate].
    gate_up_flops = GroupedGemm.flops_func(active_tokens, hidden, 2 * intermediate_size)
    # down_proj: [active_tokens, intermediate] x per-expert [intermediate, hidden].
    down_flops = GroupedGemm.flops_func(active_tokens, intermediate_size, hidden)
    flops = gate_up_flops + down_flops
    if _RT.mlp_bias:
        flops += active_tokens * (2 * intermediate_size + hidden)
    bytes_moved: float | None = None
    if bpe is not None:
        gate_up_bytes = GroupedGemm.bytes_func(
            active_tokens, hidden, 2 * intermediate_size, groups, wbpe, bpe
        )
        down_bytes = GroupedGemm.bytes_func(
            active_tokens, intermediate_size, hidden, groups, wbpe, bpe
        )
        bytes_moved = gate_up_bytes + down_bytes
        if _RT.mlp_bias:
            bytes_moved += active_tokens * (2 * intermediate_size + hidden) * bpe
        if _RT.ep > 1:
            # Dispatch sends each token's hidden state to its top_k expert ranks;
            # combine gathers the expert outputs back (MoEComm: num_tokens * hidden
            # * bpe each). Token activations move at the activation dtype, not wbpe.
            bytes_moved += 2 * active_tokens * hidden * bpe
    return "MoE", "matrix", flops, bytes_moved, tokens * hidden


def _formula_quant_rows(
    node_id: str,
    section: str,
    instances: int,
    category: str,
    role: str,
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> list[RooflineOp]:
    """Quant rows for a formula-costed GEMM or MoE that is not an nn.Linear leaf."""
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return []
    matrices: list[tuple[int, int, int]] = []
    if category == "MoE" and _RT.moe_top_k and _RT.moe_intermediate:
        intermediate = _shard(_RT.moe_intermediate, _RT.tp)
        active = _shard(tokens * _RT.moe_top_k, _RT.ep)
        experts = _shard(_RT.moe_num_experts, _RT.ep) if _RT.moe_num_experts else 0
        groups = min(experts, active) if experts else 1
        matrices = [
            (active, groups * 2 * intermediate, hidden),
            (active, groups * hidden, intermediate),
        ]
    elif category == "GEMM" and role in ("ffn", "moe") and _RT.dense_intermediate:
        intermediate = _shard(_RT.dense_intermediate, _RT.tp)
        matrices = [
            (tokens, 2 * intermediate, hidden),
            (tokens, hidden, intermediate),
        ]
    elif category == "GEMM" and role == "head":
        try:
            vocab = _shard(resolve_dim(dims[Symbol.VOCAB.value], dims), _RT.tp)
        except (KeyError, DimResolutionError):
            return []
        matrices = [(tokens, vocab, hidden)]
    if not matrices:
        return []
    return _quant_matrix_rows(
        node_id, section, instances, matrices, bpe, dtype, gpu_arch
    )


def _moe_extra_ops(
    section: str,
    node_id: str,
    instances: int,
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
    present: frozenset[str] = frozenset(),
) -> list[RooflineOp]:
    """Cost opaque MoE routing and optional shared experts.

    The selected-expert GEMMs live in :func:`_moe_cost`.  The model graph often
    collapses the router and shared-expert path into that same opaque tile, so
    publish their config-derived work as separate rows.  ``present`` names the
    pieces the graph already drew as leaves (``router``, ``shared``), which are
    not costed again.
    """
    if not (_RT.moe_num_experts and _RT.hidden_size):
        return []
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return []
    ops: list[RooflineOp] = []
    wbpe = _RT.weight_bpe if _RT.weight_bpe is not None else bpe
    if "router" not in present:
        router_flops = GEMM.flops_func(tokens, _RT.moe_num_experts, hidden, bias=False)
        router_bytes = (
            GEMM.bytes_func(
                tokens, _RT.moe_num_experts, hidden, False, wbpe, wbpe, bpe, bpe
            )
            if bpe is not None
            else None
        )
        router_elements = tokens * _RT.moe_num_experts
        ops.append(
            _analytical_op(
                f"{node_id}.router",
                section,
                "router",
                instances,
                "GEMM",
                "matrix",
                router_flops,
                router_bytes,
                router_elements,
                dtype,
                gpu_arch,
            )
        )
        ops.extend(
            _quant_matrix_rows(
                f"{node_id}.router",
                section,
                instances,
                [(tokens, _RT.moe_num_experts, hidden)],
                bpe,
                dtype,
                gpu_arch,
            )
        )
        softmax_bytes = (
            Softmax.bytes_func(tokens, _RT.moe_num_experts, bpe)
            if bpe is not None
            else None
        )
        ops.append(
            _analytical_op(
                f"{node_id}.router_softmax",
                section,
                "router_softmax",
                instances,
                "Softmax",
                "vector",
                Softmax.flops_func(tokens, _RT.moe_num_experts),
                softmax_bytes,
                router_elements,
                dtype,
                gpu_arch,
            )
        )
    if _RT.moe_top_k:
        selected = tokens * _RT.moe_top_k * hidden
        combine_bytes = 3 * selected * bpe if bpe is not None else None
        ops.append(
            _analytical_op(
                f"{node_id}.combine",
                section,
                "combine",
                instances,
                "BinaryElementwise",
                "vector",
                BinaryElementwise.flops_func(selected),
                combine_bytes,
                selected,
                dtype,
                gpu_arch,
            )
        )
    if "shared" not in present and _RT.num_shared_experts and _RT.moe_intermediate:
        shared_width = _RT.moe_intermediate * _RT.num_shared_experts
        previous = _RT.dense_intermediate
        _RT.dense_intermediate = shared_width
        try:
            shared = _dense_ffn_cost(dims, bpe)
        finally:
            _RT.dense_intermediate = previous
        if shared is not None:
            category, maf_type, flops, bytes_moved, output_elements = shared
            ops.append(
                _analytical_op(
                    f"{node_id}.shared_expert",
                    section,
                    "shared_expert",
                    instances,
                    category,
                    maf_type,
                    flops,
                    bytes_moved,
                    output_elements,
                    dtype,
                    gpu_arch,
                )
            )
            intermediate = _shard(shared_width, _RT.tp)
            ops.extend(
                _quant_matrix_rows(
                    f"{node_id}.shared_expert",
                    section,
                    instances,
                    [
                        (tokens, 2 * intermediate, hidden),
                        (tokens, hidden, intermediate),
                    ],
                    bpe,
                    dtype,
                    gpu_arch,
                )
            )
    return ops


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
    weight_bpe: float | None = None,
) -> tuple[str, str, float, float | None] | None:
    """Return ``(category, maf_type, flops, bytes)`` for an op, or ``None`` if unmodeled."""
    if computation in _GEMM_NN_MODULE_COMPUTATIONS:
        return _gemm_cost_linear(
            input_shapes,
            output_shape,
            bpe,
            weight_bpe,
            bias=_linear_has_bias(details),
        )
    if computation.lower() in _GEMM_FUNCTIONAL_LABELS:
        return _gemm_cost_matmul(input_shapes, output_shape, bpe, weight_bpe)
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
        # sqrt, rsqrt, and reciprocal count two flops per element, the same
        # weight _FLOPS_PER_ELEM in triton_compiled_perf_model.py assigns them
        # for fused Triton kernels. Other unary labels keep one flop per
        # element. The table itself is not used here: it omits many of these
        # labels and gives others different weights (e.g. exp, sigmoid, gelu).
        flops_func = (
            aten_sqrt.flops_func if label in ("sqrt", "rsqrt", "reciprocal") else None
        )
        return _unary_elementwise_cost(output_shape, bpe, flops_func)
    if _is_embedding_lookup(computation):
        return _embedding_lookup_cost(input_shapes, output_shape, bpe)
    return None


def _is_embedding_lookup(computation: str) -> bool:
    """True for an embedding gather or a tensor index that copies those rows.

    Matches module labels (``Embedding``, ``Token Embedding``, ``EmbeddingBag``)
    and the graph's indexing labels (``Indexing``, ``Index select``,
    ``aten::index``). A general ``Gather`` stays unmodeled: on a graph that
    name is overloaded and is often one element per index, which this row
    formula does not describe. Traces still cost ``aten::gather`` directly.
    """
    text = computation.lower().replace("_", " ").strip()
    labels = {
        "embedding",
        "tokenembedding",
        "embeddingbag",
        "indexing",
        "indexselect",
        "index",
    }
    squashed = text.replace(" ", "")
    if squashed in labels:
        return True
    leaf = squashed.split("::")[-1]
    return leaf in labels


def _embedding_lookup_cost(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    """Bytes for the rows the lookup gathers. The lookup itself is not a GEMM."""
    if not output_shape or bpe is None:
        return None
    gathered = prod(output_shape)
    tokens = None
    for shape in input_shapes:
        if not shape or len(shape) >= len(output_shape):
            continue
        candidate = prod(shape)
        if candidate and gathered % candidate == 0:
            tokens = candidate
            break
    if not tokens:
        tokens = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    hidden = gathered // tokens if tokens else 0
    index_bpe = 8
    flops = IndexedCopy.flops_func(tokens, hidden)
    bytes_moved = IndexedCopy.bytes_func(tokens, hidden, bpe, index_bpe)
    return IndexedCopy.category, "vector", flops, bytes_moved


def _linear_has_bias(details: str) -> bool:
    """True when a node records an explicit linear bias."""
    return bool(re.search(r"(?:^|[;,\s])bias=(?:1|true)\b", details or "", re.I))


def _gemm_cost_linear(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
    weight_bpe: float | None = None,
    bias: bool = False,
) -> tuple[str, str, float, float | None] | None:
    if not input_shapes or not input_shapes[0] or not output_shape:
        return None
    input_shape = input_shapes[0]
    k = input_shape[-1]
    n = output_shape[-1]
    m = prod(output_shape[:-1]) if len(output_shape) > 1 else 1
    flops = GEMM.flops_func(m, n, k, bias)
    wbpe = weight_bpe if weight_bpe is not None else bpe
    bytes_moved = GEMM.bytes_func(m, n, k, bias, wbpe, wbpe, bpe, bpe)
    return "GEMM", "matrix", flops, bytes_moved


def _gemm_cost_matmul(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
    weight_bpe: float | None = None,
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
    # A parameter matrix has fewer leading dimensions than the batched
    # activation it contracts with. Either operand can be that parameter.
    # Two batched operands are both activations.
    bpe_a = (
        weight_bpe
        if weight_bpe is not None and len(a_shape) < len(output_shape)
        else bpe
    )
    bpe_b = (
        weight_bpe
        if weight_bpe is not None and len(b_shape) < len(output_shape)
        else bpe
    )
    bytes_moved = GEMM.bytes_func(m, n, k, False, bpe_a, bpe_b, bpe, bpe)
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
    flops_func=None,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    nelems = prod(output_shape)
    if flops_func is None:
        flops_func = UnaryElementwise.flops_func
    flops = flops_func(nelems)
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
