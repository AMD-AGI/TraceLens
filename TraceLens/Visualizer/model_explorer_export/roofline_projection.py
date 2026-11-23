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
    over ``G`` distinct weight matrices, the experts the routed rows are expected
    to reach (``num_experts * (1 - (1 - 1/num_experts) ** active_tokens)``), so
    the weight-byte term reflects the experts actually streamed (``G`` falls back
    to 1, the prior single-weight cost, when ``num_experts`` is unknown). Tensor
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
import warnings
from pathlib import Path
from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, NamedTuple

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
    RMSNorm,
    Softmax,
    UnaryElementwise,
)
from TraceLens.PerfModel.op_flops import op_flops_per_elem
from TraceLens.PerfModel.utils import name2bpe, torch_dtype_map
from TraceLens.TreePerf.tree_perf import compute_roofline_time_and_bound
from TraceLens.Visualizer.model_explorer_export.ast_build import (
    build_model_explorer_payload,
)
from TraceLens.Visualizer.model_explorer_export.computation_roles import (
    ATTENTION_ROLE,
    ROLE_BASIS_COMPUTATION,
)
from TraceLens.Visualizer.model_explorer_export.shapes import parse_shape_dims

# Constants

ATTENTION_CATEGORY = "Attention"

# Bytes per element by torch_dtype_map's simulation-precision name.
_BYTES_PER_ELEMENT: dict[str, float] = {
    "fp64": name2bpe("fp64"),
    "fp32": name2bpe("fp32"),
    "fp16": name2bpe("fp16"),
    "bf16": name2bpe("bf16"),
    "fp8": name2bpe("fp8"),
    "fp4": name2bpe("fp4"),
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
# RMSNorm.flops_func and _rmsnorm_bytes. channel_first classes
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
        "reciprocal sqrt",
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
# Graph labels whose OP_FLOPS_PER_ELEM row is not the label itself.
_LABEL_TO_OP_FLOPS_ROW = {
    "negate": "neg",
    "reciprocal sqrt": "rsqrt",
    "cosine": "cos",
    "sine": "sin",
    "cast": "to",
    "subtract": "sub",
    "multiply": "mul",
    "divide": "div",
    "power": "pow",
    "elementwise_add": "add",
    "elementwise_mul": "mul",
    "variance": "var",
    "product": "prod",
}
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
    # Origami GEMM time, or the SDPA tile model timed with Origami. Set only
    # with --enable-origami-gemm. The roofline time stays the roof.
    origami_time_us: float | None = None

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

    @property
    def total_origami_time_us(self) -> float | None:
        return (
            None
            if self.origami_time_us is None
            else self.origami_time_us * self.instances
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
    kv_cache_bpe: float | None = None,
    enable_origami_gemm: bool = False,
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
    _begin_projection(spec, kv_cache_bpe, enable_origami_gemm)
    moe_top_k = spec.num_experts_per_tok
    moe_intermediate_size = spec.moe_intermediate_size or spec.intermediate_size
    nodes = _projection_graph_nodes(spec, checkpoint)

    result = RooflineProjection(batch_size=batch_size, seq_len=seq_len)
    run = _ProjectionRun(
        nodes=nodes,
        dims=dims,
        gpu_arch=gpu_arch,
        moe_top_k=moe_top_k,
        moe_intermediate_size=moe_intermediate_size,
        opaque_attention=_opaque_attention_scopes(nodes),
        replaced_eager_attention=set(),
    )
    for node in nodes:
        graph_op = _graph_op(node)
        if graph_op is not None:
            result.ops.extend(_graph_op_rows(run, graph_op))
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


def _begin_projection(
    spec: ArchitectureSpec, kv_cache_bpe: float | None, enable_origami_gemm: bool
) -> None:
    """Reset the module-level state one :func:`project_roofline` call reads."""
    global _origami_enabled, _origami_arch, _origami_arch_warned, _sim_note
    _populate_introspection(spec)
    # K and V are read from the cache. ``None`` keeps them at the activation width.
    _RT.kv_cache_bpe = kv_cache_bpe
    # One flag, matching the perf report's Origami GEMM switch: GEMMs call
    # Origami, and attention calls the SDPA tile model with Origami timing
    # each tile GEMM.
    _origami_enabled = enable_origami_gemm
    _origami_arch = None
    _origami_arch_warned = False
    _sim_note = None


def _projection_graph_nodes(
    spec: ArchitectureSpec, checkpoint: str | Path | None
) -> list[Any]:
    """Nodes of the merged graph, expanded only down to roofline-formula ops."""
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
    return graphs[0]["nodes"] if graphs else []


class _ProjectionRun(NamedTuple):
    """Inputs one :func:`project_roofline` call shares across its nodes."""

    nodes: list[Any]
    dims: dict[str, Any]
    gpu_arch: dict[str, Any] | None
    moe_top_k: int | None
    moe_intermediate_size: int | None
    opaque_attention: set[str]
    # Scopes whose eager softmax chain is already priced as one SDPA row.
    replaced_eager_attention: set[str]


class _GraphOp(NamedTuple):
    """A drawn compute node: its identity plus the :func:`_parse_op_node` fields."""

    node: dict[str, Any]
    node_id: str
    namespace: str
    instances: int
    section: str
    computation: str
    class_name: str
    operation: str
    output_text: str
    input_texts: list[list[str]]
    role: str
    details: str


def _graph_op(node: Any) -> _GraphOp | None:
    """*node* as a :class:`_GraphOp`, or ``None`` when it computes nothing."""
    if not isinstance(node, dict):
        return None
    parsed = _parse_op_node(node)
    if parsed is None:
        return None
    namespace = str(node.get("namespace") or "")
    instances = _repeat_count(namespace, node)
    return _GraphOp(node, str(node.get("id")), namespace, instances, *parsed)


def _has_computation_role(node: dict[str, Any]) -> bool:
    """The node's role was read from its own computation, not its module."""
    return _attr_text(node.get("attrs"), "role_basis") == ROLE_BASIS_COMPUTATION


def _opaque_attention_scopes(nodes: list[Any]) -> set[str]:
    """Ids of the opaque attention blocks in *nodes*.

    An opaque attention block (role attention, not one step of the softmax)
    already prices the whole layer. The eager matmul chain under it must not
    be priced again. A chain with no such block is the eager expansion of
    softmax attention, whose shapes stop at S and miss the KV cache.
    """
    scopes: set[str] = set()
    for node in nodes:
        graph_op = _graph_op(node)
        if graph_op is None or _has_computation_role(node):
            continue
        if graph_op.role == "attention" and not _is_gated_delta(graph_op.computation):
            scopes.add(graph_op.node_id)
    return scopes


def _is_skipped_body(graph_op: _GraphOp) -> bool:
    """A reference-body op whose work the kernel or a fused epilogue does."""
    # The conv call is the kernel. Its reference body (slice, silu) is not
    # a second pass over the activation.
    if _RT.sconv_kernel and _is_expanded_conv_body(graph_op.node_id):
        return True
    # No attention mask is passed, so this function returns the activation
    # unchanged. The multiply and the cast are inside that branch.
    if _is_padding_mask_body(graph_op.node_id):
        return True
    # fp8 SwiGLU stores neither the SiLU result nor the bf16 product.
    # down_proj's activation quant reads both branches.
    return _RT.weight_bpe == 1 and _is_mlp_activation_interior(
        graph_op.namespace, graph_op.node_id, graph_op.computation
    )


def _graph_op_rows(run: _ProjectionRun, graph_op: _GraphOp) -> list[RooflineOp]:
    """The rows one drawn compute node adds to the projection."""
    if _is_skipped_body(graph_op):
        return []
    # A leaf whose role was read from its own computation is one step of
    # that computation (the score matmul, the softmax, the value matmul).
    # Each is costed as the op it is. The boundary formulas below price a
    # whole block, so handing a step the block's role would charge the full
    # attention once per step.
    role, computation_role = graph_op.role, ""
    if _has_computation_role(graph_op.node):
        computation_role, role = role, ""
    op = _project_op(
        graph_op.node_id,
        graph_op.section,
        graph_op.computation,
        graph_op.class_name,
        graph_op.operation,
        graph_op.output_text,
        graph_op.input_texts,
        run.dims,
        graph_op.instances,
        run.gpu_arch,
        run.moe_top_k,
        run.moe_intermediate_size,
        role,
        graph_op.details,
        computation_role=computation_role,
    )
    if _node_is_draft_container(graph_op.node):
        # The synthetic stack, or the container's own drawn children, costs
        # this module. The container node itself is not a second copy.
        return []
    if _is_eager_softmax_step(graph_op.node_id, graph_op.computation, computation_role):
        return _eager_attention_rows(run, graph_op)
    _dims_only, dtype, bpe = _output_dtype_bpe(graph_op.output_text)
    rows = _attention_block_rows(run, graph_op, op, role, bpe, dtype)
    rows.extend(_moe_block_rows(run, graph_op, op, role, bpe, dtype))
    rows.extend(_weight_quant_rows(run, graph_op, op, role, bpe, dtype))
    return rows


def _eager_attention_rows(run: _ProjectionRun, graph_op: _GraphOp) -> list[RooflineOp]:
    """One SDPA row for the first step of an eager attention chain, else none."""
    scope = _attention_scope(graph_op.node_id)
    if scope in run.opaque_attention or scope in run.replaced_eager_attention:
        return []
    run.replaced_eager_attention.add(scope)
    _dims_only, dtype, bpe = _output_dtype_bpe(graph_op.output_text)
    boundary = _softmax_attention_row(
        scope,
        graph_op.section,
        graph_op.instances,
        run.dims,
        bpe,
        dtype,
        run.gpu_arch,
        heads=_eager_qk_heads(graph_op.output_text, graph_op.input_texts, run.dims),
    )
    return [boundary] if boundary is not None else []


def _attention_block_rows(
    run: _ProjectionRun,
    graph_op: _GraphOp,
    op: RooflineOp,
    role: str,
    bpe: float | None,
    dtype: str | None,
) -> list[RooflineOp]:
    """*op*, split per layer variant or followed by an opaque block's extra ops."""
    if not (role == "attention" and op.modeled and op.category == "Attention"):
        return [op]
    if _attention_variants_cover(op.instances):
        return _attention_variant_rows(
            op,
            graph_op.node_id,
            graph_op.section,
            run.nodes,
            graph_op.node,
            run.dims,
            bpe,
            dtype,
            run.gpu_arch,
        )
    if _attention_has_projection_leaves(run.nodes, graph_op.node):
        return [op]
    # A fused attention leaf is costed at only its SDPA core. Its packed
    # projections, indexer, and norms live inside the opaque block.
    return [
        op,
        *_attention_extra_ops(
            graph_op.section,
            graph_op.node_id,
            graph_op.instances,
            run.dims,
            bpe,
            dtype,
            run.gpu_arch,
            owner_class=graph_op.class_name,
        ),
    ]


def _moe_block_rows(
    run: _ProjectionRun,
    graph_op: _GraphOp,
    op: RooflineOp,
    role: str,
    bpe: float | None,
    dtype: str | None,
) -> list[RooflineOp]:
    """The router and shared-expert rows of an opaque MoE block."""
    if not (
        role in ("ffn", "moe")
        and op.modeled
        and op.category == "MoE"
        and not _is_leaf_computation(graph_op.computation)
    ):
        return []
    return _moe_extra_ops(
        graph_op.section,
        graph_op.node_id,
        graph_op.instances,
        run.dims,
        bpe,
        dtype,
        run.gpu_arch,
        present=_moe_leaves_present(run.nodes, graph_op.node),
    )


def _weight_quant_rows(
    run: _ProjectionRun,
    graph_op: _GraphOp,
    op: RooflineOp,
    role: str,
    bpe: float | None,
    dtype: str | None,
) -> list[RooflineOp]:
    """Quantization side rows of a modeled GEMM or MoE op with quantized weights."""
    if not op.modeled or op.category not in ("GEMM", "MoE"):
        return []
    if _RT.weight_bpe is None:
        return []
    if op.category == "GEMM" and graph_op.computation in _GEMM_NN_MODULE_COMPUTATIONS:
        return _quant_side_ops(
            graph_op.node_id,
            graph_op.section,
            graph_op.instances,
            graph_op.input_texts,
            graph_op.output_text,
            run.dims,
            bpe,
            dtype,
            run.gpu_arch,
        )
    return _formula_quant_rows(
        graph_op.node_id,
        graph_op.section,
        graph_op.instances,
        op.category,
        role,
        run.dims,
        bpe,
        dtype,
        run.gpu_arch,
    )


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
        bpe = (
            _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
        ) or _RT.act_bpe
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

    # fp8 weights are read by the GEMM itself (aiter/ck block-scale). A separate
    # dequant would materialize a bf16 copy that those kernels do not write.
    # Narrower schemes (fp4, int4) still expand the weight before the multiply.
    if _RT.weight_bpe != 1:
        _unary("weight_dequant", weight_elements, _RT.weight_bpe, bpe)
    if _RT.weight_bpe == 1:
        # SwiGLU's down projection consumes gate and up. The quant kernel
        # reads both and writes the packed activation once.
        reads = act_elements * (2 if _swiglu_down_projection(node_id) else 1)
        op = RooflineOp(
            name=f"{node_id}.act_quant",
            section=section,
            computation="act_quant",
            operation="",
            category=None,
            modeled=False,
            reason=None,
            output_elements=act_elements,
            instances=instances,
        )
        ops.append(
            _finish_op(
                op,
                "UnaryElementwise",
                "vector",
                UnaryElementwise.flops_func(act_elements),
                reads * bpe + act_elements * _RT.weight_bpe,
                dtype,
                gpu_arch,
            )
        )
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
    computation_role: str = "",
) -> RooflineOp:
    base = _unmodeled_op(node_id, section, computation, operation, instances)
    dims_only, dtype, bpe = _output_dtype_bpe(output_text)
    ctx = _OpContext(base, dims, dims_only, dtype, bpe, gpu_arch)

    # Introspective boundaries: the modeling source tags each module with a role
    # ("attention"/"ffn") and the arch spec says whether the model is MoE, so we
    # cost the custom attention and routed-expert blocks without matching names.
    op = None
    if role == "attention":
        op = _attention_role_op(ctx, computation, input_texts)
    elif role in ("ffn", "moe"):
        op = _ffn_role_op(ctx, role, class_name, computation)
    if op is None:
        op = _module_boundary_op(
            ctx, computation, class_name, moe_top_k, moe_intermediate_size
        )
    if op is not None:
        return op
    return _dispatch_leaf_op(ctx, computation, input_texts, details, computation_role)


def _unmodeled_op(
    name: str, section: str, computation: str, operation: str, instances: int
) -> RooflineOp:
    """A row with no cost yet, for a cost path to fill or explain."""
    return RooflineOp(
        name=name,
        section=section,
        computation=computation,
        operation=operation,
        category=None,
        modeled=False,
        reason=None,
        output_elements=None,
        instances=instances,
    )


def _output_dtype_bpe(output_text: str) -> tuple[list[str], str | None, float | None]:
    """``(output dims, dtype, bytes per element)`` of a drawn output shape.

    A shape without a dtype, or with one that has no known width, is costed at
    the activation width.
    """
    dims_only, dtype = _split_shape_text(output_text)
    bpe = (
        _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
    ) or _RT.act_bpe
    return dims_only, dtype, bpe


class _OpContext(NamedTuple):
    """One node's unmodeled row and the inputs every cost path of
    :func:`_project_op` shares."""

    base: RooflineOp
    dims: dict[str, Any]
    dims_only: list[str]
    dtype: str | None
    bpe: float | None
    gpu_arch: dict[str, Any] | None


def _finish_drawn(
    ctx: _OpContext, cost: tuple[str, str, float, float | None]
) -> RooflineOp:
    """Finish *cost*, counting output elements from the drawn output shape."""
    category, maf_type, flops, bytes_moved = cost
    try:
        oshape = resolve_shape(ctx.dims_only, ctx.dims)
        ctx.base.output_elements = prod(oshape) if oshape else 1
    except DimResolutionError:
        ctx.base.output_elements = None
    return _finish_op(
        ctx.base, category, maf_type, flops, bytes_moved, ctx.dtype, ctx.gpu_arch
    )


def _finish_counted(
    ctx: _OpContext, cost: tuple[str, str, float, float | None, int]
) -> RooflineOp:
    """Finish a *cost* that carries its own output element count."""
    category, maf_type, flops, bytes_moved, output_elements = cost
    ctx.base.output_elements = output_elements
    return _finish_op(
        ctx.base, category, maf_type, flops, bytes_moved, ctx.dtype, ctx.gpu_arch
    )


def _attention_role_op(
    ctx: _OpContext, computation: str, input_texts: list[list[str]]
) -> RooflineOp | None:
    """A whole attention block, or ``None`` to fall through to the other paths.

    A gated-delta block always stops here, unmodeled unless it is the
    recurrent update.
    """
    if _is_gated_delta(computation):
        # gdn_attention_core prices the fused recurrent update. The chunked
        # kernels are a different algorithm and have no performance model.
        # The SDPA boundary below is the full-attention layer.
        if _is_recurrent_gated_delta(computation):
            gdn = _gated_delta_cost(input_texts, ctx.dims, ctx.bpe)
            if gdn is not None:
                return _finish_drawn(ctx, gdn)
        return ctx.base
    att = _attention_boundary_cost(ctx.dims, ctx.bpe)
    if att is not None:
        return _finish_drawn(ctx, att)
    return None


def _ffn_role_op(
    ctx: _OpContext, role: str, class_name: str, computation: str
) -> RooflineOp | None:
    """A routed-expert or opaque dense FFN block, or ``None`` when neither fits."""
    if _RT.moe_top_k and _looks_like_moe(class_name):
        moe = _moe_cost(ctx.dims, _RT.moe_top_k, _RT.moe_intermediate, ctx.bpe)
        if moe is not None:
            return _finish_counted(ctx, moe)
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
        dense = _dense_ffn_cost(ctx.dims, ctx.bpe)
        if dense is not None:
            return _finish_counted(ctx, dense)
    return None


def _module_boundary_op(
    ctx: _OpContext,
    computation: str,
    class_name: str,
    moe_top_k: int | None,
    moe_intermediate_size: int | None,
) -> RooflineOp | None:
    """A short conv or an engine (vLLM/SGLang) module, or ``None``."""
    if _is_short_conv(computation, class_name):
        conv = _short_conv_cost(ctx.dims, ctx.dims_only, ctx.bpe)
        if conv is not None:
            return _finish_counted(ctx, conv)
    engine_cost = _engine_boundary_cost(
        class_name, ctx.dims, moe_top_k, moe_intermediate_size, ctx.bpe
    )
    if engine_cost is not None:
        return _finish_counted(ctx, engine_cost)
    return None


def _dispatch_leaf_op(
    ctx: _OpContext,
    computation: str,
    input_texts: list[list[str]],
    details: str,
    computation_role: str,
) -> RooflineOp:
    """Cost a leaf op from its resolved shapes through the PerfModel dispatch."""
    base = ctx.base
    try:
        output_shape = resolve_shape(ctx.dims_only, ctx.dims)
        base.output_elements = prod(output_shape) if output_shape else 1
    except DimResolutionError as exc:
        base.reason = f"unresolved output dim {exc}"
        return base

    try:
        input_shapes = [
            resolve_shape(shape, ctx.dims) if shape else None for shape in input_texts
        ]
        cost = _dispatch_op_cost(
            computation,
            input_shapes,
            output_shape,
            ctx.bpe,
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
    if computation_role == ATTENTION_ROLE:
        # The cost is the op's own; only its family changes. The score and value
        # matmuls multiply two activations, so they run at the activation
        # precision and need no weight-quantization side rows, which a GEMM
        # label would have given them.
        category = ATTENTION_CATEGORY
    return _finish_op(
        base, category, maf_type, flops, bytes_moved, ctx.dtype, ctx.gpu_arch
    )


def _finish_op(
    base: RooflineOp,
    category: str,
    maf_type: str,
    flops: float,
    bytes_moved: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
) -> RooflineOp:
    sim = _take_sim()
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
    base.origami_time_us = _origami_time_us(
        category, precision, dtype, bytes_moved, gpu_arch, sim
    )
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
        "config_attn_heads",
        "config_attn_kv_heads",
        "config_attn_head_dim",
        "kv_cache_bpe",
        "act_bpe",
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
        self.config_attn_heads = None
        self.config_attn_kv_heads = None
        self.config_attn_head_dim = None
        self.kv_cache_bpe = None
        self.act_bpe = None


_RT = _RuntimeIntrospection()

# ``--enable-origami-gemm`` on the roofline projection. Times GEMMs with
# Origami and attention with the SDPA tile model. Off unless the caller sets it.
_origami_enabled = False
# Shape note for the next ``_finish_op``. A list of ("gemm", (M, N, K, B))
# or one ("sdpa", {B, H_Q, N_Q, N_KV, d_h, bytes}).
_sim_note: list | None = None
_origami_arch: dict[str, Any] | None = None
_origami_arch_warned = False


def _set_sim(note: list | None) -> None:
    global _sim_note
    _sim_note = note


def _take_sim() -> list | None:
    global _sim_note
    note = _sim_note
    _sim_note = None
    return note


def _origami_arch_for(gpu_arch: dict[str, Any] | None) -> dict[str, Any] | None:
    """Arch dict Origami and the SDPA tile model can time against.

    Bundled specs name the peaks and the bandwidth. Origami also needs the
    CU count, LDS, and clock, which a trace's deviceProperties would supply
    to the perf report. With no trace, the installed GPU fills the gaps.
    """
    global _origami_arch, _origami_arch_warned
    if gpu_arch is None:
        return None
    if _origami_arch is not None:
        return _origami_arch
    arch = dict(gpu_arch)
    missing = [
        key
        for key in ("num_cus", "lds_capacity", "freq_mhz", "gemm_units_per_cu")
        if not arch.get(key)
    ]
    if missing:
        try:
            import origami

            hardware = origami.get_hardware_for_device(0)
        except Exception as exc:  # noqa: BLE001
            if not _origami_arch_warned:
                warnings.warn(
                    "Origami GEMM and SDPA times need num_cus, lds_capacity, "
                    f"freq_mhz, and gemm_units_per_cu ({exc}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _origami_arch_warned = True
        else:
            arch.setdefault("num_cus", int(hardware.N_CU))
            arch.setdefault("lds_capacity", int(hardware.lds_capacity))
            arch.setdefault(
                "freq_mhz", int(round(float(hardware.compute_clock_ghz) * 1000))
            )
            parallel = getattr(hardware, "parallel_mi_cu", None)
            if parallel:
                arch.setdefault("gemm_units_per_cu", int(parallel))
    _origami_arch = arch
    return arch


def _origami_time_us(
    category: str,
    precision: str | None,
    dtype: str | None,
    bytes_moved: float | None,
    gpu_arch: dict[str, Any] | None,
    sim: list | None,
) -> float | None:
    """Origami time for one launch, or None when the flag is off or it cannot time this op.

    GEMM uses ``gemm_time_us``. Attention uses the SDPA tile model, which
    times each tile GEMM with Origami. A sum of several GEMMs (a dense
    SwiGLU tile) adds their times. Faster than the roofline is still
    recorded; the plot keeps the roofline unless this time is slower.
    """
    if not _origami_enabled or not sim:
        return None
    arch = _origami_arch_for(gpu_arch)
    if arch is None:
        return None
    from TraceLens.PerfModel import origami_helper, sdpa_tile

    total = 0.0
    timed = False
    for kind, payload in sim:
        try:
            if kind == "gemm" and category == "GEMM" and precision:
                m, n, k, batch = payload
                part = origami_helper.gemm_time_us(
                    arch, int(m), int(n), int(k), int(batch), precision
                )
            elif kind == "sdpa" and category in {ATTENTION_CATEGORY, "SDPA"}:
                part = sdpa_tile.sdpa_fwd_time_us(
                    arch,
                    precision,
                    dtype or precision,
                    payload.get("bytes", bytes_moved),
                    payload["B"],
                    payload["H_Q"],
                    payload["N_Q"],
                    payload["N_KV"],
                    payload["d_h"],
                    fa=bool(arch.get("l1_bw_gbps")),
                    gemm_time=origami_helper.gemm_time_us,
                )
            else:
                part = None
        except (KeyError, ValueError, TypeError):
            part = None
        if part is not None:
            total += float(part)
            timed = True
    return total if timed else None


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
    _RT.act_bpe = _activation_bpe(spec)[0]
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
    # Kept from the checkpoint. A hybrid model resolves a different head count
    # in the linear-attention class, and that single constructor value must not
    # become the softmax attention's width.
    _RT.config_attn_heads = _RT.attn_heads
    _RT.config_attn_kv_heads = _RT.attn_kv_heads
    _RT.config_attn_head_dim = _RT.attn_head_dim
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
        tcfg,
        "sconv_kernel_size",
        "conv_kernel_size",
        "short_conv_kernel_size",
        "linear_conv_kernel_dim",
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
    text = f"{computation} {class_name}".lower().replace("_", "").replace(" ", "")
    if "conv2d" in text or "conv3d" in text:
        return False
    return any(
        token in text for token in ("shortconv", "sconv", "causalconv", "conv1d")
    )


def _is_expanded_conv_body(node_id: str) -> bool:
    """Reference body of ``causal_conv1d_fn``. The call site is the kernel."""
    return "causal_conv1d_fn" in node_id


def _is_padding_mask_body(node_id: str) -> bool:
    """Body of ``apply_mask_to_padding_states``.

    The function multiplies by the mask only when one is passed. A shape-only
    projection has no mask, and that multiply is not HBM traffic.
    """
    return "apply_mask_to_padding_states" in node_id


def _is_mlp_activation_interior(namespace: str, node_id: str, computation: str) -> bool:
    """SiLU and the gate×up multiply inside an MLP.

    With fp8 weights the activation quant on ``down_proj`` is that epilogue:
    it reads both GEMM outputs and writes the packed activation. The SiLU
    result and the bf16 product are not stored. The node id carries the
    ``mlp`` segment; the namespace alone is the parent section.
    """
    text = f"{namespace}/{node_id}".lower()
    if "/mlp/" not in text:
        return False
    return computation.lower() in {"silu", "gelu", "relu", "multiply", "mul"}


def _swiglu_down_projection(node_id: str) -> bool:
    """The third SwiGLU GEMM, whose input is the product of the two branches."""
    return "down_proj" in node_id.lower()


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


def _is_gated_delta(computation: str) -> bool:
    return "gated_delta" in (computation or "").lower()


def _is_recurrent_gated_delta(computation: str) -> bool:
    """The fused per-token update that ``gdn_attention_core`` models.

    The chunked prefill kernels share the ``gated_delta`` name and run a
    different algorithm, with no performance model of their own.
    """
    lowered = (computation or "").lower()
    return "recurrent" in lowered and "gated_delta" in lowered


def _attention_scope(node_id: str) -> str:
    """The attention module a step belongs to.

    Expanded steps are ``.../self_attn/seq:15:@attention:@op_...``. The marker
    is ``@attention``, not a path segment, so every step of one module shares
    one scope and is priced once.
    """
    head, sep, _tail = node_id.partition("@attention")
    return head.rstrip("/:") if sep else node_id


def _is_eager_softmax_step(
    node_id: str, computation: str, computation_role: str
) -> bool:
    """A matmul/softmax that is one step of an expanded attention, not the block."""
    return (
        computation_role == ATTENTION_ROLE
        and "@attention" in node_id
        and not _is_gated_delta(computation)
    )


def _gated_delta_cost(
    input_texts: list[list[str]],
    dims: dict[str, Any],
    bpe: float | None,
) -> tuple[str, str, float, float | None] | None:
    """One recurrent gated-delta layer from its own q/k/v shapes.

    This is ``gdn_attention_core``: the fused update whose state stays in
    registers, at ``7 · d_v · d_k`` FLOPs per token and value head. The stopped
    kernel is called with q, k, v as ``[B, S, H, D]``, but the drawn node also
    carries the packed projection and the per-head gates, so the value tensor
    is the rank-4 input with the most heads. Projections are separate GEMMs.
    """
    from TraceLens.PerfModel.extensions.attention_perf_model_extensions import (
        gdn_attention_core,
    )

    try:
        rank4 = [
            shape
            for text in input_texts
            if text and (shape := resolve_shape(text, dims)) and len(shape) == 4
        ]
    except DimResolutionError:
        return None
    if not rank4:
        return None
    q = rank4[0]
    v = max(rank4, key=lambda shape: shape[-2])
    tokens = prod(q[:-2]) if len(q) > 2 else q[0]
    h_v = v[-2]
    d_v = v[-1]
    d_k = q[-1]
    if tokens <= 0 or h_v <= 0 or d_v <= 0 or d_k <= 0:
        return None
    flops = float(gdn_attention_core._gdn_flops_func(h_v, d_k, d_v, tokens))
    bytes_moved = None
    if bpe is not None:
        bytes_moved = float(
            gdn_attention_core._gdn_bytes_func(h_v, d_k, d_v, tokens, bpe)
        )
    return ATTENTION_CATEGORY, "matrix", flops, bytes_moved


def _eager_qk_heads(
    output_text: str,
    input_texts: list[list[str]],
    dims: dict[str, Any],
) -> tuple[int, int] | None:
    """``(query heads, head dim)`` from an expanded QK matmul, when it has them.

    The query is ``[B, H, S, D]``. A hybrid checkpoint's linear attention
    publishes a different head count, so the score matmul is the one that
    says how wide this softmax layer is.
    """
    try:
        shapes = [resolve_shape(text, dims) for text in input_texts[:2] if text]
    except DimResolutionError:
        return None
    for shape in shapes:
        if (
            shape
            and len(shape) == 4
            and shape[-1] != shape[-2]
            and shape[-1] > 1
            and shape[-3] > 1
        ):
            return int(shape[-3]), int(shape[-1])
    return None


def _softmax_attention_row(
    name: str,
    section: str,
    instances: int,
    dims: dict[str, Any],
    bpe: float | None,
    dtype: str | None,
    gpu_arch: dict[str, Any] | None,
    heads: tuple[int, int] | None = None,
) -> RooflineOp | None:
    """One SDPA layer in place of the eager QK/softmax/AV chain.

    That chain's key length is the query length, so a decode step (S=1) costs
    nothing and a prefill is the full non-causal score matrix. These layers
    are the full-attention variant, so the KV length is the whole cache and
    the head count is this matmul's, not the linear attention's.
    """
    try:
        b = resolve_dim(dims["B"], dims)
        s = resolve_dim(dims["S"], dims)
    except (KeyError, DimResolutionError):
        return None
    past = _resolve_past(dims)
    n_kv = past + s
    if heads is not None:
        h_q, d_h = heads
    else:
        h_q = _RT.config_attn_heads or _RT.attn_heads
        d_h = _RT.config_attn_head_dim or _RT.attn_head_dim
    h_kv = _RT.config_attn_kv_heads or _RT.attn_kv_heads or h_q
    if not h_q or not d_h or not h_kv:
        att = _attention_boundary_cost(dims, bpe)
        if att is None:
            return None
        category, maf_type, flops, bytes_moved = att
    else:
        h_q = _shard_heads(h_q, _RT.tp)
        h_kv = _shard_heads(h_kv, _RT.tp)
        causal = bool(_RT.causal_attention and past == 0 and s == n_kv)
        flops, bytes_moved = _sdpa_flops_bytes(
            b, s, h_q, n_kv, h_kv, d_h, d_h, causal, bpe
        )
        flops = float(flops)
        bytes_moved = float(bytes_moved) if bpe is not None else None
        category, maf_type = ATTENTION_CATEGORY, "matrix"
        _set_sdpa_sim(b, h_q, s, n_kv, d_h, bytes_moved)
    row = _unmodeled_op(name, section, "sdpa", "", instances)
    return _finish_op(row, category, maf_type, flops, bytes_moved, dtype, gpu_arch)


def _sdpa_flops_bytes(
    b: int,
    s: int,
    h_q: int,
    n_kv: float,
    h_kv: int,
    d_qk: int,
    d_v: int,
    causal: bool,
    bpe: float | None,
) -> tuple[Any, Any]:
    """``SDPA`` FLOPs and bytes (``None`` without *bpe*), K/V at the cache width."""
    flops = SDPA.flops_func(b, s, h_q, n_kv, h_kv, d_qk, d_v, causal)
    bytes_moved = (
        SDPA.bytes_func(
            b,
            s,
            h_q,
            n_kv,
            h_kv,
            d_qk,
            d_v,
            causal,
            bpe,
            kv_bytes_per_element=_RT.kv_cache_bpe,
        )
        if bpe is not None
        else None
    )
    return flops, bytes_moved


def _set_sdpa_sim(
    b: int, h_q: int, n_q: int, n_kv: int, d_h: int, bytes_moved: float | None
) -> None:
    """Leave the SDPA tile shape for the next :func:`_finish_op`."""
    _set_sim(
        [
            (
                "sdpa",
                {
                    "B": b,
                    "H_Q": h_q,
                    "N_Q": n_q,
                    "N_KV": n_kv,
                    "d_h": d_h,
                    "bytes": bytes_moved,
                },
            )
        ]
    )


def _attention_boundary_cost(dims, bpe, window=_WINDOW_UNSET):
    """Roofline cost of ONE attention layer, introspected from the arch spec
    (query/kv heads, head dim, sliding window, local/global layer mix). KV length
    is averaged across the local (windowed) and global layers so the per-layer
    cost times the layer-repeat count reproduces the true aggregate."""
    _set_sim(None)
    if not (_RT.attn_heads and _RT.attn_head_dim):
        return None
    try:
        b = resolve_dim(dims["B"], dims)
        s = resolve_dim(dims["S"], dims)
    except (KeyError, DimResolutionError):
        return None
    past = _resolve_past(dims)
    ctx = past + s
    h_q, h_kv, d_qk, d_v = _attention_widths()
    eff_n_kv = _effective_kv_len(ctx, window)
    # ``SDPA`` only accepts causal masking when query and key/value lengths
    # agree.  During warm-cache decode Q is a suffix of KV and the engine
    # applies the model's causal policy in its cache kernel; its FLOP count is
    # unchanged by the mask.  Cold/prefill self-attention therefore takes the
    # model-declared causal path instead of the historical forced ``False``.
    causal = _RT.causal_attention and past == 0 and s == eff_n_kv
    flops, bytes_moved = _sdpa_flops_bytes(
        b, s, h_q, eff_n_kv, h_kv, d_qk, d_v, causal, bpe
    )
    _set_sdpa_sim(b, h_q, s, int(round(eff_n_kv)), d_qk, bytes_moved)
    return "Attention", "matrix", flops, bytes_moved


def _attention_widths() -> tuple[int, int, int, int]:
    """Per-GPU ``(h_q, h_kv, d_qk, d_v)`` of one attention layer."""
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
    return h_q, h_kv, d_qk, d_v


def _effective_kv_len(ctx: int, window: Any) -> float:
    """KV length one layer attends to out of *ctx*.

    An explicit *window* (``None`` for full attention) wins. Otherwise the
    length is averaged over the layer variants, or over the local and global
    layers.
    """
    if window is not _WINDOW_UNSET:
        return min(ctx, window) if isinstance(window, int) and window > 0 else ctx
    if _RT.variant_windows:
        return _variant_mean_kv_len(ctx)
    return _local_global_mean_kv_len(ctx)


def _variant_mean_kv_len(ctx: int) -> float:
    """KV length averaged over the per-variant windows, weighted by layer count."""
    total = sum(count for _label, count, _window in _RT.variant_windows) or 1
    weighted = 0.0
    for _label, count, variant_window in _RT.variant_windows:
        length = (
            min(ctx, variant_window)
            if isinstance(variant_window, int) and variant_window > 0
            else ctx
        )
        weighted += count * length
    return weighted / total


def _local_global_mean_kv_len(ctx: int) -> float:
    """KV length averaged over the sliding-window and global layers."""
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
    return (n_local * kv_local + n_global * ctx) / n_layers if n_layers else ctx


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
        _set_sim([("gemm", (tokens, n, k, 1))])
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
                    RMSNorm.flops_func(elements, d_h, True),
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
    _set_sim(
        [
            ("gemm", (tokens, 2 * intermediate, hidden, 1)),
            ("gemm", (tokens, hidden, intermediate, 1)),
        ]
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
    norm_flops = RMSNorm.flops_func(norm_elems, hidden, True)
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
        norm_flops = RMSNorm.flops_func(norm_elems, hidden, True)
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


def _experts_touched(experts: int, routed_rows: int) -> float:
    """Expected distinct experts that ``routed_rows`` rows reach.

    A row picks an expert uniformly, so an expert is missed with probability
    ``(1 - 1/experts) ** routed_rows``. A few rows reach about one expert each;
    many rows reach them all. Charging every expert whenever ``routed_rows``
    exceeds the expert count would read weights no row selected: 256 rows over
    128 experts reach about 111, not 128.
    """
    if experts <= 0 or routed_rows <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - 1.0 / experts) ** routed_rows)


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
    ``G`` distinct expert weight matrices -- the experts ``active_tokens`` routed
    rows are expected to reach under uniform routing, see
    :func:`_experts_touched` -- rather than a single shared weight. FLOPs are unchanged (only the selected token-expert
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
    groups = _experts_touched(experts_here, active_tokens) if experts_here else 1
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
    """Quant rows for a formula-costed GEMM that is not an nn.Linear leaf."""
    if category == "MoE":
        # A grouped expert GEMM reads its packed (e.g. MXFP4) weights directly,
        # as the aiter/cktile and Triton MoE kernels do. No separate dequant pass
        # runs beside it, so the expert weight bytes come from _moe_cost alone.
        return []
    try:
        tokens = resolve_dim(dims["B"], dims) * resolve_dim(dims["S"], dims)
        hidden = resolve_dim(dims[Symbol.HIDDEN.value], dims)
    except (KeyError, DimResolutionError):
        return []
    matrices: list[tuple[int, int, int]] = []
    if category == "GEMM" and role in ("ffn", "moe") and _RT.dense_intermediate:
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
    _set_sim(None)
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
        return _reduce_cost(
            input_shapes, output_shape, bpe, _label_flops_per_element(label)
        )
    if label in _BINARY_POINTWISE_COMPUTATIONS:
        return _binary_elementwise_cost(
            input_shapes, output_shape, bpe, _label_flops_per_element(label)
        )
    if label in _UNARY_POINTWISE_COMPUTATIONS:
        return _unary_elementwise_cost(
            output_shape, bpe, _label_flops_per_element(label)
        )
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
    _set_sim([("gemm", (m, n, k, 1))])
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
    _set_sim([("gemm", (m, n, k, 1))])
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
    bpe = (
        _BYTES_PER_ELEMENT.get(torch_dtype_map(dtype)) if dtype else None
    ) or _RT.act_bpe
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
        SDPA.bytes_func(
            b,
            n_q,
            h_q,
            n_kv,
            h_kv,
            d_h_qk,
            d_h_v,
            causal,
            bpe,
            kv_bytes_per_element=_RT.kv_cache_bpe,
        )
        if bpe is not None
        else None
    )
    _set_sim(
        [
            (
                "sdpa",
                {
                    "B": b,
                    "H_Q": h_q,
                    "N_Q": n_q,
                    "N_KV": n_kv,
                    "d_h": d_h_qk,
                    "bytes": bytes_moved,
                },
            )
        ]
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
        flops = RMSNorm.flops_func(num_elems, num_channels, is_affine)
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


def _label_flops_per_element(label: str) -> int:
    """Per-element flops of a pointwise or reduction label, from ``OP_FLOPS_PER_ELEM``."""
    return op_flops_per_elem(_LABEL_TO_OP_FLOPS_ROW.get(label, label))


def _unary_elementwise_cost(
    output_shape: tuple[int, ...],
    bpe: float | None,
    flops_per_elem: int,
) -> tuple[str, str, float, float | None] | None:
    if not output_shape:
        return None
    nelems = prod(output_shape)
    flops = UnaryElementwise.flops_func(nelems, flops_per_elem)
    bytes_moved = (
        UnaryElementwise.bytes_func(nelems, bpe, bpe) if bpe is not None else None
    )
    return "UnaryElementwise", "vector", flops, bytes_moved


def _binary_elementwise_cost(
    input_shapes: list[tuple[int, ...] | None],
    output_shape: tuple[int, ...],
    bpe: float | None,
    flops_per_elem: int,
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
    flops = BinaryElementwise.flops_func(nelems_out, flops_per_elem)
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
    flops_per_elem: int,
) -> tuple[str, str, float, float | None] | None:
    if not input_shapes or not input_shapes[0]:
        return None
    input_shape = input_shapes[0]
    num_input_elems = prod(input_shape)
    num_output_elems = prod(output_shape) if output_shape else 1
    flops = Reduce.flops_func(num_input_elems, flops_per_elem)
    bytes_moved = (
        Reduce.bytes_func(num_input_elems, num_output_elems, bpe, bpe)
        if bpe is not None
        else None
    )
    return "Reduce", "vector", flops, bytes_moved
