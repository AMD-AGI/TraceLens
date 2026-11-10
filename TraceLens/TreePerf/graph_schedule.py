###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TreePerf launch list of a DeepSeek-V4 graph, with the input shapes of each launch.

A replayed graph is one run of the SGLang DeepSeek-V4 forward at one token
count. That forward is one attention block and one MoE block per layer. The
checkpoint says which attention block (its ``compress_ratios``) and which MoE
block (the first ``num_hash_layers`` layers route by hash). Walking that list
in order and pairing it with the graph's kernels in order gives every matched
kernel the tensors it was called with.

The device symbol (``Cijk_…``, ``ck::kernel_gemm_…``) is the library's choice
for that call on this GPU. The launch records the call and the shapes.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field, replace

from TraceLens.TraceUtils.utils.annotation_utils import parse_annotation


@dataclass(frozen=True)
class GraphModel:
    """The dimensions of the forward the graph covers, on one GPU."""

    hidden_size: int
    q_lora_rank: int
    o_lora_rank: int
    num_attention_heads: int
    head_dim: int
    o_groups: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    compress_ratios: tuple[int, ...]
    num_hash_layers: int
    vocab_size: int
    tp: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # The checkpoint's rotary dim. Zero means the full head dim.
    qk_rope_head_dim: int = 0
    hc_mult: int = 1
    n_shared_experts: int = 0
    # ``fp4`` selects the packed expert layout. Anything else keeps the
    # logical routed GEMM, which is what a non-FP4 expert runs.
    expert_dtype: str = ""
    # ``quantization_config.weight_block_size``, as ``(block_n, block_k)``.
    weight_block_size: tuple[int, ...] = ()
    max_position_embeddings: int = 0
    # CUDA-graph capture buckets. Replay pads a request up to the smallest
    # bucket that covers it. Empty means the graph was captured at the
    # request size.
    capture_buckets: tuple[int, ...] = ()
    # ``sliding_window``: the keys each query attends in the local window.
    sliding_window: int = 0
    # The device the graph ran on. The mHC split-K policy is keyed by both.
    # Zero compute units leaves the split width out of the launch shapes.
    num_cus: int = 0
    gfx_arch: str = ""
    # Numbers and switch defaults the engine's source fixes (see
    # ``engine_facts``), as ``(name, value)`` pairs. A fact that is absent
    # leaves the launch shapes that depend on it out.
    facts: tuple[tuple[str, object], ...] = ()
    # Every process group the trace recorded (``tp``, ``moe_ep``, ``moe_dp``,
    # ``attn_cp``, ``pp``, ...). Empty means only ``tp`` shards, at ``self.tp``.
    groups: dict[str, int] = field(default_factory=dict)

    def fact(self, name: str, default=None):
        return dict(self.facts).get(name, default)

    def degree(self, name: str) -> int:
        """The size of one parallelism axis on this GPU's run.

        ``moe_ep`` is the expert-parallel group, ``moe_tp`` the tensor
        parallel group inside an expert, ``attn_tp`` the attention group,
        ``attn_cp`` context parallel, ``moe_dp`` expert data parallel, ``pp``
        pipeline. A group the trace did not create is 1, except the tensor
        axes, which fall back to ``tp``.
        """
        from TraceLens.ModelUtils.parallel_plan import ParallelDegree

        recorded = dict(self.groups)
        recorded.setdefault("tp", self.tp)
        return ParallelDegree(tp=self.tp, groups=recorded).size(name)

    def __post_init__(self) -> None:
        if self.tp < 1:
            raise ValueError(f"tp must be >= 1, got {self.tp}")
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                "compress_ratios must cover every layer, "
                f"got {len(self.compress_ratios)} for {self.num_hidden_layers} layers"
            )


@dataclass(frozen=True)
class Invocation:
    """One model run. ``tokens`` is the M dimension of the graph's GEMMs.

    ``prefill`` selects the extend graph. Its indexer runs once per chunk of
    the query, and its GEMMs are the prefill kernels rather than the decode
    ones.
    """

    batch_size: int
    seq_len: int
    past_len: int
    tokens: int
    prefill: bool = False


@dataclass(frozen=True)
class Launch:
    """One device launch in the graph, in the order the forward issues it."""

    op_name: str
    key: str
    input_shapes: tuple[tuple[int, ...], ...]
    layer: int | None = None
    # Dtypes in argument order, as the profiler records them on the captured
    # call. The roofline reads these, not only the shapes.
    input_types: tuple[str, ...] = ()
    # Scalar arguments, aligned with ``input_shapes``. ``""`` is a tensor.
    concrete: tuple[str, ...] = ()
    # Stride of each argument. Empty means the tensor is contiguous.
    input_strides: tuple[tuple[int, ...], ...] = ()
    # This kernel is launched by the same CPU call as the previous launch.
    # A split-K reduce and a schedule's second kernel are that case: the
    # collector keeps them on the call, not as a second op.
    same_call: bool = False


def graph_model_from_config(
    config: dict, tp: int, groups: dict | None = None
) -> GraphModel:
    """Build a :class:`GraphModel` from a Hugging Face ``config.json``.

    ``groups`` is every process-group size the trace recorded. Tensor parallel
    is also ``tp``, so a caller that only knows that degree still shards.
    """
    ratios = config.get("compress_ratios") or ()
    quant = config.get("quantization_config") or {}
    block = quant.get("weight_block_size") or ()
    buckets = config.get("capture_buckets") or ()
    return GraphModel(
        hidden_size=int(config["hidden_size"]),
        q_lora_rank=int(config["q_lora_rank"]),
        o_lora_rank=int(config["o_lora_rank"]),
        num_attention_heads=int(config["num_attention_heads"]),
        head_dim=int(config["head_dim"]),
        o_groups=int(config["o_groups"]),
        moe_intermediate_size=int(config["moe_intermediate_size"]),
        n_routed_experts=int(config["n_routed_experts"]),
        num_experts_per_tok=int(config["num_experts_per_tok"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        compress_ratios=tuple(int(r) for r in ratios),
        num_hash_layers=int(config.get("num_hash_layers") or 0),
        vocab_size=int(config["vocab_size"]),
        tp=tp,
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        qk_rope_head_dim=int(config.get("qk_rope_head_dim") or 0),
        hc_mult=int(config.get("hc_mult") or 1),
        n_shared_experts=int(config.get("n_shared_experts") or 0),
        expert_dtype=str(config.get("expert_dtype") or ""),
        weight_block_size=tuple(int(size) for size in block),
        max_position_embeddings=int(config.get("max_position_embeddings") or 0),
        capture_buckets=tuple(int(size) for size in buckets),
        sliding_window=int(config.get("sliding_window") or 0),
        groups={**dict(groups or {}), "tp": int((groups or {}).get("tp", tp))},
    )


def tp_from_trace(metadata: dict | None) -> int | None:
    """Tensor-parallel degree recorded on the trace's ``distributedInfo``.

    The ``tp`` process group (``tp:device`` and its ``tp:cpu`` twin). A trace
    whose only world has one rank is not sharded. A trace with no such group,
    or whose device and cpu groups disagree, does not say. Every other group
    is :func:`parallel_degree_from_trace`.
    """
    from TraceLens.ModelUtils.parallel_plan import parallel_degree_from_trace

    degree = parallel_degree_from_trace(metadata)
    if degree is None:
        return None
    if "tp" in degree.groups:
        return degree.groups["tp"]
    if degree.groups.get("world") == 1:
        return 1
    return None


def num_cus_from_trace(metadata: dict | None) -> int | None:
    """Compute units recorded on the trace's ``deviceProperties``.

    PyTorch writes one entry per visible GPU, and ``numSms`` is the
    compute-unit count on these ROCm traces. The count is returned when every
    GPU agrees. A missing field or a mixed node returns ``None``, and the
    caller keeps the architecture table.
    """
    if not metadata:
        return None
    devices = metadata.get("deviceProperties")
    if not isinstance(devices, list) or not devices:
        return None
    counts = set()
    for device in devices:
        if not isinstance(device, dict) or "numSms" not in device:
            return None
        counts.add(int(device["numSms"]))
    if len(counts) != 1:
        return None
    count = counts.pop()
    return count if count > 0 else None


def replay_size(size: int, buckets: tuple[int, ...]) -> int:
    """The captured size a replay of ``size`` uses.

    The graph is the smallest bucket that covers the request. With no buckets
    the request was captured as itself.
    """
    covering = [bucket for bucket in buckets if bucket >= size]
    return min(covering) if covering else size


def invocation_from_annotation(name: str) -> Invocation | None:
    """Batch, sequence length, and past length recorded on a step annotation.

    Decode query length is one token per request. The past length is the mean
    cached length, ``g_sk / batch``, because the annotation stores the sum.
    Prefill tokens are the context sum, and the past length is zero.
    """
    ann = parse_annotation(name)
    if not ann.matched:
        return None
    if ann.context_requests and not ann.generation_requests:
        batch = ann.context_requests or ann.batch_size or 1
        tokens = ann.context_sum or ann.c_sq or batch
        seq = tokens // batch if batch else tokens
        # c_sk sums the KV length. Whatever exceeds the query is already cached.
        kv = ann.c_sk or tokens
        past = max(kv - tokens, 0) // batch if batch else 0
        return Invocation(batch, seq, past, tokens, prefill=True)
    batch = ann.generation_requests or ann.batch_size or 1
    tokens = ann.g_sq or batch
    past = (ann.g_sk // batch) if batch and ann.g_sk else 0
    return Invocation(batch, 1, past, tokens)


def _kv_span(past_len: int, seq_len: int, ratio: int) -> int:
    total = past_len + seq_len
    if ratio > 1:
        return max(total // ratio, 1)
    return max(total, 1)


def _matches(kernel_name: str, key: str) -> bool:
    n = kernel_name
    if key == "mhc_pre_rms":
        return "mhc_pre_big_fuse_rmsnorm" in n
    if key == "mhc_sqrsum":
        return "mhc_pre_gemm_sqrsum" in n
    if key == "mhc_pre":
        return "mhc_pre_big_fuse_kernel" in n and "rmsnorm" not in n
    if key == "mhc_fused":
        return "mhc_fused_post_pre" in n
    if key == "mhc_post":
        return "mhc_post_kernel" in n
    if key == "quant":
        return "dynamic_per_group_scaled_quant" in n
    if key == "gemm_a8w8":
        return "gemm_a8w8_blockscale_preshuffle" in n
    if key == "gemm_red":
        return "gemm_a8w8_blockscale_reduce" in n
    if key == "rms_quant":
        return "fused_rms_fp8_group_quant" in n
    if key == "ck_gemm":
        return "kernel_gemm_xdl" in n
    if key == "qk_rope":
        return "fuse_qk_norm_rope" in n
    if key == "tensile":
        return n.startswith("Cijk_") or "Cijk_" in n
    if key == "c128":
        return "flash_c128" in n
    if key == "c4":
        return "flash_c4" in n
    if key == "flashmla":
        return "fused_norm_rope_flashmla" in n
    if key == "mla":
        return "mla_a8w8" in n or "opus_mla" in n
    if key == "attn_asm":
        return "_fwd_kernel_stage2" in n
    if key == "rope":
        return "apply_rotary_emb" in n
    if key == "wo_quant":
        return "_wo_a_quant" in n
    if key == "gemm_mx":
        return "gemm_a8w8_mxscale" in n
    if key == "hgemm":
        return "hgemm_" in n
    if key == "topk":
        return "topk_gating_kernel" in n
    if key == "hash_topk":
        return "hash_topk" in n
    if key == "shared":
        return "fused_append_shared" in n
    if key == "moe_sort":
        return "moe_sorting" in n
    if key == "mx_sort":
        return "fused_mx_quant_moe_sort" in n
    if key == "moe1":
        return "mfma_moe1" in n
    if key == "moe_red":
        return "moe_reduction" in n
    if key == "moe2":
        return "mfma_moe2" in n or "opus_moe_stage2" in n
    if key == "allreduce":
        return "cross_device_reduce" in n or "quickreduce::allreduce" in n
    if key == "hc_head":
        return "_hc_head_kernel" in n
    if key == "embed":
        return "vocab_parallel_embedding" in n
    if key == "rope_had":
        return "rope_hadamard_rotate" in n and "norm_rope" not in n
    if key == "norm_rope_had":
        return "norm_rope_hadamard" in n
    if key == "indexer":
        return "pa_mqa_logits" in n
    if key == "topk_main":
        return "topk_main_kernel" in n
    if key == "fill_tail":
        return "fill_compress_tail" in n
    if key == "ck_tile":
        return "ck_tile" in n
    if key == "w128":
        return "write_c128" in n
    if key == "w4":
        return "write_c4" in n
    if key == "pf_len":
        return "_prefill_lengths_kernel" in n
    if key == "pf_idx":
        return "_build_prefill_indices_kernel" in n
    if key == "swa":
        return "_swa_scatter_kernel" in n
    if key == "gemm_sf":
        return "gemm_a8w8_scale_preload" in n
    if key == "gate":
        # The prefill gate GEMM is hgemm at short M and a Tensile kernel at long M.
        return "hgemm_" in n or "Cijk_" in n
    if key == "opus_sort":
        return "opus_moe_sorting" in n
    if key == "mxfp4":
        return "mxfp4_moe_sort" in n
    if key == "pf_prep":
        return "_prefill_schedule_prep_kernel" in n
    if key == "pf_cta":
        return "_prefill_cta_info_kernel" in n
    if key == "init_meta":
        return "_init_compressed_attn_metadata_kernel" in n
    if key == "cta_info":
        return "_decode_cta_info_kernel" in n
    if key == "sched_prep":
        return "_decode_schedule_prep_kernel" in n
    if key == "paged_idx":
        return "_v4_paged_decode_indices_kernel" in n
    if key == "plan":
        return "plan_compress_decode_kernel" in n
    if key == "pad_pt":
        return "_pad_page_table_kernel" in n
    if key == "rmsnorm":
        return "rmsnorm_quant_kernel" in n
    if key == "allgather":
        return "allgather_vec" in n
    return _torch_kernel(n, key)


_TORCH_KERNELS = {
    "t_arange": "arange_cuda_out",
    "t_cmp": "compare_scalar_kernel",
    "t_eq": "CompareEqFunctor",
    "t_mfill": "masked_fill_kernel",
    "t_index": "index_kernel_impl",
    "t_clamp": "launch_clamp_scalar",
    "t_fdiv": "div_floor_kernel_cuda",
    "t_fill": "FillFunctor",
    "t_scan": "rocprim",
    "t_mul": "MulFunctor",
    "t_rem": "remainder_kernel_cuda",
    "t_and": "BitwiseAndFunctor",
    "t_where": "where_kernel_impl",
}


def _torch_kernel(name: str, key: str) -> bool:
    """The eager torch kernels the replay's metadata prep launches."""
    if key == "copy":
        # A copy into a contiguous slice (one row) is a device memcpy, otherwise
        # an elementwise kernel.
        return "copy_kernel_cuda" in name or "copyBuffer" in name
    if key == "t_add":
        return "CUDAFunctor" in name and "_add<" in name
    return key in _TORCH_KERNELS and _TORCH_KERNELS[key] in name
    return False


def align_kernels(
    kernel_names: list[str], launches: list[Launch]
) -> list[Launch | None]:
    """Pair kernels with launches in order.

    A kernel that is not the next launch (an indexing or copy kernel between
    two launches) is left unmatched. The launch cursor only advances on a hit,
    so the next real launch still receives its own shapes.
    """
    assigned: list[Launch | None] = [None] * len(kernel_names)
    cursor = 0
    for index, name in enumerate(kernel_names):
        if cursor < len(launches) and _matches(name, launches[cursor].key):
            assigned[index] = launches[cursor]
            cursor += 1
        elif cursor and _follows(name, launches[cursor - 1], assigned[index - 1]):
            assigned[index] = launches[cursor - 1]
    return assigned


def _index_cast(launch: Launch | None) -> bool:
    """An int64 ``aten::copy_`` the index op launches to cast its indices."""
    return (
        launch is not None
        and launch.key == "copy"
        and bool(launch.input_types)
        and launch.input_types[0] == "long int"
    )


def group_stopped_ops(
    assigned: list[Launch | None],
) -> list[tuple[int, list[int]]]:
    """Group kernels the collector keeps on one CPU op.

    Each entry is ``(owner, members)``. The owner is the launch whose name
    and argument shapes the row carries. Members are indexes into
    ``assigned``, in launch order.
    """
    count = len(assigned)
    owner = list(range(count))
    for index, launch in enumerate(assigned):
        if launch is None or launch.key != "t_index":
            continue
        cursor = index - 1
        while cursor >= 0 and _index_cast(assigned[cursor]) and owner[cursor] == cursor:
            owner[cursor] = index
            cursor -= 1
    for index in range(1, count):
        previous = assigned[index - 1]
        launch = assigned[index]
        if previous is None or launch is None:
            continue
        # ``same_call`` is the schedule's second launch of one CPU call.
        # ``launch is previous`` is a kernel the call launches that the
        # schedule did not list on its own (the scan's state kernel, a
        # split-K epilogue, the stage-2 reduction of an atomic dispatch
        # that still emitted one).
        if launch.same_call or launch is previous:
            owner[index] = owner[index - 1]
    groups: list[tuple[int, list[int]]] = []
    current: int | None = None
    members: list[int] = []
    for index in range(count):
        if current is None or owner[index] != current:
            if members:
                groups.append((current, members))
            current = owner[index]
            members = [index]
        else:
            members.append(index)
    if members and current is not None:
        groups.append((current, members))
    return groups


def _follows(name: str, launch: Launch, previous: Launch | None) -> bool:
    """A second kernel of the launch just before it.

    A long-K Tensile GEMM splits K across workgroups and a PostGSU kernel
    reduces the partials. A long scan runs a state kernel ahead of the scan.
    A stage-2 MoE call reduces its partials in ``moe_reduction`` when the
    dispatch did not accumulate them itself.
    """
    if previous is not launch:
        return False
    if launch.key in ("tensile", "gate"):
        return "PostGSU" in name
    if launch.key == "moe2":
        # The stage-2 call reduces its partials in a second kernel when the
        # dispatch does not accumulate them itself. An atomic tuned row omits
        # that launch; a graph that still runs the kernel keeps it on the call.
        return "moe_reduction" in name
    return launch.key == "t_scan" and "rocprim" in name


# Dtypes of each launch, in the argument order the captured call records.
# The roofline indexes these positions; a launch keeps the prefix that fits
# the tensors it actually passes.
_BF16 = "c10::BFloat16"
_FP8 = "c10::Float8_e4m3fn"
_FP4 = "c10::Float4_e2m1fn_x2"
_E8M0 = "c10::Float8_e8m0fnu"
# c10 names the roofline's byte table already understands. ``seq_lens`` and the
# request slots are int64; the HIP backend's ``cuda_int32_kwargs`` is int32.
_I32 = "int"
_I64 = "long int"
_SCALAR = "Scalar"
# ``CompressorDecodePlan.plan_d`` is a ``[batch, 16]`` uint8 DecodePlan.
# ``view(int32)`` makes four words per row.
_DECODE_PLAN_BYTES = 16
_DTYPES = {
    "gemm_a8w8": (_FP8, _FP8, "float", "float"),
    "gemm_red": (_FP8, _FP8, "float", "float"),
    "ck_gemm": (_FP8, _FP8, "float", "float"),
    "ck_tile": (_FP8, _FP8, "float", "float"),
    "gemm_mx": (_FP8, _FP8, _BF16, "unsigned char", "unsigned char"),
    "gemm_sf": (_FP8, _FP8, _BF16, "unsigned char", "unsigned char"),
    "hgemm": (_BF16, _BF16),
    "tensile": (_BF16, _BF16),
    "gate": (_BF16, _BF16),
    # fused_moe_: hidden, packed w1, packed w2, weights, ids, then the scales.
    "moe1": (_BF16, _FP4, _FP4, "float", "int", _E8M0, _E8M0),
    "moe2": (_BF16, _FP4, _FP4, "float", "int", _E8M0, _E8M0),
    "moe_red": (_BF16, _FP4, _FP4, "float", "int", _E8M0, _E8M0),
    "rmsnorm": (_BF16, _BF16, _BF16),
    # mla_decode_v4_asm: q, rope q, kv, kv rope, the index arrays, then the
    # four scalars (max_seqlen_q, softmax scale, out_16_nosplit, num_kv_splits),
    # the split buffers, and the valid-split scalar.
    "mla": (
        _FP8,
        _BF16,
        _FP8,
        _BF16,
        "int",
        "int",
        "int",
        "int",
        "float",
        "Scalar",
        "Scalar",
        "Scalar",
        "Scalar",
        "float",
        "float",
        _BF16,
        "int",
        "Scalar",
        "",
    ),
    # dynamic_per_group_scaled_quant(out, input, scales).
    "quant": (_FP8, _BF16, "float"),
    "wo_quant": (_FP8, _BF16, "float"),
    # fused_dynamic_mx_quant_moe_sort_hip(out, scales, x, ...): x is bf16.
    "mx_sort": (_FP4, _E8M0, _BF16, "int", "int", "float"),
    "moe_sort": ("int", "int", "int", "int", _BF16),
}


def _row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Contiguous strides of ``shape``. An empty shape is a scalar argument."""
    stride = [1] * len(shape)
    step = 1
    for index in range(len(shape) - 2, -1, -1):
        step *= shape[index + 1]
        stride[index] = step
    return tuple(stride)


def _expand_stride(
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    target: tuple[int, ...],
) -> tuple[int, ...]:
    """Stride of ``tensor.expand(target)``, from ``inferExpandGeometry``.

    A new leading axis takes the span of the axis after it, and an axis that
    grows from one element is a broadcast (stride 0).
    """
    expanded = [0] * len(target)
    sizes = [0] * len(target)
    tensor_dim = len(shape)
    for index in range(len(target) - 1, -1, -1):
        offset = len(target) - 1 - index
        dim = tensor_dim - 1 - offset
        size = shape[dim] if dim >= 0 else 1
        step = stride[dim] if dim >= 0 else sizes[index + 1] * expanded[index + 1]
        sizes[index] = target[index]
        expanded[index] = step if size == target[index] else 0
    return tuple(expanded)


def _is_contiguous(shape: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """PyTorch contiguity: an axis of length 0 or 1 does not have to step."""
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size <= 1:
            continue
        if step != expected:
            return False
        expected *= size
    return True


def _dtype_cast_stride(
    shape: tuple[int, ...], stride: tuple[int, ...]
) -> tuple[int, ...]:
    """``.to(dtype)`` keeps the stride when the source is already contiguous.

    A single-element column is contiguous at any stride, so the cast of a
    one-row plan column keeps the column step. A longer column is packed.
    """
    if _is_contiguous(shape, stride):
        return stride
    return _row_major(shape)


def _unfold_repeat(
    shape: tuple[int, ...], repeat: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shape and destination stride of the copy inside ``Tensor.repeat``.

    The repeat reference unfolds a contiguous buffer of the repeated shape and
    copies the expanded input into that view, then permutes the view back.
    """
    padded = (1,) * (len(repeat) - len(shape)) + tuple(shape)
    target = tuple(size * times for size, times in zip(padded, repeat))
    ur_shape = list(target)
    ur_stride = list(_row_major(target))
    for dim, dim_size in enumerate(padded):
        step = max(dim_size, 1)
        ur_shape.append(dim_size)
        ur_stride.append(ur_stride[dim])
        ur_shape[dim] = (ur_shape[dim] - dim_size) // step + 1
        ur_stride[dim] *= step
    return tuple(ur_shape), tuple(ur_stride)


def launch_arguments(launch: Launch) -> dict:
    """The argument record a captured call of ``launch`` would carry.

    The roofline reads ``Input Dims``, ``Input type`` and ``Input Strides``
    from the CPU op. A replayed graph has no CPU op, so the launch supplies
    the same three fields.
    """
    count = len(launch.input_shapes)
    if launch.input_strides:
        strides = [list(stride) for stride in launch.input_strides]
    else:
        strides = [list(_row_major(shape)) for shape in launch.input_shapes]
    record = {
        "Input Dims": [list(shape) for shape in launch.input_shapes],
        "Input type": list(launch.input_types[:count]),
        "Input Strides": strides,
    }
    if launch.concrete:
        record["Concrete Inputs"] = list(launch.concrete)
    return record


def _launch(
    op_name: str,
    key: str,
    shapes: tuple[tuple[int, ...], ...],
    layer: int | None,
    types: tuple[str, ...] | None = None,
    concrete: tuple[str, ...] = (),
    strides: tuple[tuple[int, ...], ...] = (),
    same_call: bool = False,
) -> Launch:
    chosen = _DTYPES.get(key, ()) if types is None else types
    return Launch(
        op_name,
        key,
        shapes,
        layer,
        chosen[: len(shapes)],
        concrete,
        strides,
        same_call,
    )


# FlyDSL packs two FP4 values into one stored element and scales every 32 of
# them. Both numbers are the loader's ``fp4_block_k`` packing, not a tile
# chosen for one trace.
_FP4_PACK = 2
_FP4_GROUP = 32


def _aiter_weight(m: int, k: int, n: int, block: tuple[int, ...]) -> tuple:
    """``Y[M, N] = X[M, K] @ W[N, K].T``, plus the block scales when known.

    The weight argument is the parameter ``(out, in)``. The scales are
    ``(M, ceil(K / block_k))`` and ``(ceil(N / block_n), ceil(K / block_k))``.
    """
    shapes = ((m, k), (n, k))
    if len(block) < 2:
        return shapes
    block_n, block_k = block[0], block[1]
    return shapes + (
        (m, math.ceil(k / block_k)),
        (math.ceil(n / block_n), math.ceil(k / block_k)),
    )


def _aten_weight(m: int, k: int, n: int) -> tuple:
    """``aten::mm`` takes the weight already transposed, ``(K, N)``."""
    return ((m, k), (k, n))


def _quant_shapes(model: GraphModel, tokens: int, width: int) -> tuple:
    """``dynamic_per_token_scaled_quant`` as the profiler records the call.

    The fp8 output is ``[T, K]``. The bf16 input is viewed by the group, so it
    is ``[T * K / group, group]``. The scale is ``[T, K / group]``. The
    roofline reads that order: output, input, scales.
    """
    if len(model.weight_block_size) < 2:
        return ((tokens, width),)
    group = model.weight_block_size[1]
    groups = math.ceil(width / group)
    return ((tokens, width), (tokens * groups, group), (tokens, groups))


def _compressor_width(ratio: int, head_dim: int) -> int:
    """Output width of ``Compressor.wkv_gate``: ``2 * coff * head_dim``.

    ``coff`` is ``1 + overlap`` and the ratio-4 compressor overlaps its windows.
    """
    coff = 1 + (ratio == 4)
    return 2 * coff * head_dim


def _indexer_rotary_shapes(model: GraphModel, tokens: int) -> tuple:
    """``aiter.rope_rotate_activation`` on the indexer's ``[T, H, D]`` query.

    The rotary slice, the FP4 scale the kernel fills, the query, the cosine
    and sine caches, and the positions. Without the scale layout the engine
    source gives, only the tensors before it are known, so none are returned.
    """
    scale = model.fact("indexer_q_scale_shape")
    rope = model.qk_rope_head_dim or model.head_dim
    if scale is None or not model.max_position_embeddings:
        return ((tokens, model.index_n_heads, model.index_head_dim),)
    heads = model.index_n_heads
    table = (model.max_position_embeddings, rope // 2)
    return (
        (tokens, heads, rope),
        (tokens, *scale),
        (tokens, heads, model.index_head_dim),
        table,
        table,
        (tokens,),
    )


def _ck_gemm(model: GraphModel, m: int, k: int, n: int, layer: int | None) -> Launch:
    """The CK block-scale GEMM. Without a block size it stays an ``aten::mm``."""
    if model.weight_block_size:
        return _launch(
            "aiter::gemm_a8w8_blockscale_bpreshuffle",
            "ck_gemm",
            _aiter_weight(m, k, n, model.weight_block_size),
            layer,
        )
    return _launch("aten::mm", "ck_gemm", _aten_weight(m, k, n), layer)


def _wo_a_shapes(
    model: GraphModel, tokens: int, groups: int, width: int, rank: int
) -> tuple:
    """``wo_a`` argument tensors, one group at a time.

    Activation ``[T, G, D]``, weight ``[G, R, D]``, output ``[T, G, R]``,
    then the activation scale ``[T, G, D/block]`` and the weight scale
    ``[G, R/block, D/block]``.
    """
    shapes = (
        (tokens, groups, width),
        (groups, rank, width),
        (tokens, groups, rank),
    )
    if len(model.weight_block_size) < 2:
        return shapes
    block_n, block_k = model.weight_block_size[0], model.weight_block_size[1]
    return shapes + (
        (tokens, groups, math.ceil(width / block_k)),
        (groups, math.ceil(rank / block_n), math.ceil(width / block_k)),
    )


def _rope_shapes(model: GraphModel, tokens: int, heads: int) -> tuple:
    """The rotary slice and, when the checkpoint gives it, the position cache."""
    dim = model.qk_rope_head_dim or model.head_dim
    shapes = ((tokens, heads, dim),)
    if model.max_position_embeddings:
        shapes += ((model.max_position_embeddings, dim // 2), (tokens,))
    return shapes


def _shuffled_scale(rows: int, cols: int) -> tuple[int, int]:
    """The view ``shuffle_scale`` returns on the non-interleave path.

    Rows round up to a multiple of 256 and columns to a multiple of 8. The
    permutation that follows does not change that view.
    """
    return ((rows + 255) // 256 * 256, (cols + 7) // 8 * 8)


def _resident_experts(model: GraphModel) -> int:
    """Routed experts that live on this GPU, plus the shared expert.

    Expert parallel splits the routed set. The shared expert is replicated,
    so it stays on every rank. The gate still scores the full routed set.
    """
    routed = model.n_routed_experts // model.degree("moe_ep")
    return routed + model.n_shared_experts


def _moe_signature(model: GraphModel, tokens: int, hidden: int, scales: bool = False):
    """The fused MoE call, or ``None`` when the experts are not FP4.

    FP4 experts are stored packed: ``w13`` is ``(E, 2 * inter_local, hidden/2)``
    and ``w2`` is ``(E, hidden, inter_local/2)``. ``E`` is the routed experts
    on this GPU (the expert-parallel group) plus the shared one, and
    ``inter_local`` is the intermediate sharded by the expert tensor-parallel
    group. The top-k width includes that shared expert.
    """
    if model.expert_dtype != "fp4":
        return None
    inter = model.moe_intermediate_size // model.degree("moe_tp")
    experts = _resident_experts(model)
    topk = model.num_experts_per_tok + model.n_shared_experts
    shapes = (
        (tokens, hidden),
        (experts, 2 * inter, hidden // _FP4_PACK),
        (experts, hidden, inter // _FP4_PACK),
        (tokens, topk),
        (tokens, topk),
    )
    if not scales:
        return shapes
    # shuffle_scale pads a 2D scale to a multiple of 256 rows and 8 columns.
    # w13's scale is (E * 2 * inter, hidden / group); w2's is (E * hidden, inter / group).
    w13_rows, w13_cols = experts * 2 * inter, hidden // _FP4_GROUP
    w2_rows, w2_cols = experts * hidden, inter // _FP4_GROUP
    return shapes + (
        _shuffled_scale(w13_rows, w13_cols),
        _shuffled_scale(w2_rows, w2_cols),
    )


def _moe_block_m(model: GraphModel, tokens: int, topk: int, experts: int, inter: int):
    """The tile rows the MoE sorter pads each expert to.

    The dispatch decides it (the tuned row's ``block_m`` or the heuristic tile).
    Without dispatch facts it is ``get_block_size_M``'s search, which keeps the supported size whose GEMM grid fills the compute
    units in the fewest rounds, then wastes the fewest slots in the last one.
    """
    dispatch = _moe_dispatch(model, tokens)
    if dispatch is not None:
        return dispatch[0]
    num_cu = model.num_cus
    if num_cu < 1:
        return None
    tile_n = 128
    tiles_n = (inter + tile_n - 1) // tile_n
    best = None
    for size in (32, 64, 128):
        padded = tokens * topk + experts * size - topk
        grid = tiles_n * (padded + size - 1) // size
        rounds = (grid + num_cu - 1) // num_cu
        wasted = num_cu - grid % num_cu
        if best is None or (rounds, wasted) < best[0]:
            best = ((rounds, wasted), size)
    return best[1]


def _moe_sort_shapes(model: GraphModel, tokens: int, hidden: int, stage: int) -> tuple:
    """``fused_dynamic_mx_quant_moe_sort_hip(out, scales, x, ids, valid, ..., weights)``.

    ``ids`` and ``weights`` hold one slot per routed token plus the padding
    every expert gets up to the sort tile: ``T * topk + E * block - topk``.
    ``scales`` is that count rounded up to 32 rows by the scale columns padded
    to a multiple of 8. Stage 1 quantizes the hidden state and stage 2 the
    ``T * topk`` intermediate rows.
    """
    if model.expert_dtype != "fp4":
        return None
    topk = model.num_experts_per_tok + model.n_shared_experts
    experts = _resident_experts(model)
    inter = model.moe_intermediate_size // model.degree("moe_tp")
    block = _moe_block_m(model, tokens, topk, experts, inter)
    if block is None:
        return None
    slots = tokens * topk + experts * block - topk
    rows, width = (tokens, hidden) if stage == 1 else (tokens * topk, inter)
    scale = _shuffled_scale(slots, math.ceil(width / _FP4_GROUP))
    scale = ((slots + 31) // 32 * 32, scale[1])
    return ((rows, width), scale, (rows, width), (slots,), (2,), (slots,))


def _padded_moe_tokens(tokens: int, tiers: tuple[int, ...]) -> int:
    """``get_padded_M``: the token count aiter's tuned tables are keyed on.

    Below the first tier the count rounds up to a power of two. At or above a
    tier it is the largest tier that does not exceed it.
    """
    if tokens < tiers[0]:
        return 1 if tokens <= 1 else 1 << (tokens - 1).bit_length()
    return max(tier for tier in tiers if tokens >= tier)


_ROW_STAGE_PREFIXES = ("flydsl_", "opus_moe1_")


def _moe_dispatch(model: GraphModel, tokens: int) -> tuple[int, str] | None:
    """``(sort block_m, stage-2 kernel)`` as ``get_2stage_cfgs`` resolves them.

    A tuned row is used when its first stage is a FlyDSL or Opus kernel or its
    second stage is FlyDSL. Any other row (a CK-tile first stage with an Opus
    second stage) is dropped and the heuristic FlyDSL pick takes over: its tile
    rows follow the token count and its second stage accumulates atomically.
    """
    tiers = model.fact("moe_padded_tiers")
    if not tiers:
        return None
    padded = _padded_moe_tokens(tokens, tiers)
    for token, block_m, first, second in model.fact("moe_tuned_rows") or ():
        if token == padded and (
            first.startswith(_ROW_STAGE_PREFIXES) or second.startswith("flydsl_")
        ):
            return block_m, second
    tiles = model.fact("moe_fallback_tiles")
    mode = model.fact("moe_fallback_stage2_mode")
    if not tiles or not mode:
        return None
    tile = next(tile for bound, tile in tiles if bound is None or tokens < bound)
    return tile, f"flydsl_moe2_{mode}"


def _stage2_kernel(model: GraphModel, tokens: int) -> str | None:
    """The second MoE GEMM aiter selects for this token count."""
    dispatch = _moe_dispatch(model, tokens)
    return dispatch[1] if dispatch else None


def _stage2_reduces(model: GraphModel, tokens: int) -> bool:
    """Whether a separate ``moe_reduction`` follows the second GEMM.

    The atomic variants accumulate into the output themselves. A kernel the
    table does not name keeps the reduction.
    """
    kernel = _stage2_kernel(model, tokens)
    return kernel is None or "atomic" not in kernel


def _opus_stage2_shapes(model: GraphModel, tokens: int, hidden: int):
    """``opus_moe_stage2_a8w4_decode_fwd`` as the custom op records it.

    The sorted ids and weights hold ``T * topk + E * block - topk`` slots, the
    expert ids one entry per sort block. The wrapper pads the activation scale
    to ``2 * mfma_m`` rows and both scales to the kernel's column pack
    (``scale_cols_for_effective_inter_dim``). A custom op whose first parameter
    is not the ``Tensor`` class is called with a dummy tensor first.
    """
    contract = model.fact("opus_stage2_contract")
    if not contract or model.expert_dtype != "fp4":
        return None
    topk = model.num_experts_per_tok + model.n_shared_experts
    experts = _resident_experts(model)
    inter = model.moe_intermediate_size // model.degree("moe_tp")
    block = _moe_block_m(model, tokens, topk, experts, inter)
    if block is None:
        return None
    packed = contract["fp4_values_per_byte"]
    k_step = contract["bk_logical"] // packed
    if inter % k_step:
        return None
    columns = (inter // k_step + 1) // 2 * contract["scale_groups_per_row_pack"]
    row_pack = 2 * contract["mfma_m"]
    slots = tokens * topk + experts * block - topk
    rows = (slots + row_pack - 1) // row_pack * row_pack
    shapes = (
        (tokens, topk, inter),
        (experts, hidden, inter // packed),
        (rows, columns),
        (experts * hidden, columns),
        (slots,),
        (slots,),
        ((slots + block - 1) // block,),
        (2,),
        (tokens, hidden),
    )
    return ((1,),) + shapes if model.fact("opus_stage2_dummy_arg") else shapes


def _stage2_launch(model, t, h, scaled, layer, routed, inter) -> Launch:
    kernel = _stage2_kernel(model, t)
    opus = (
        _opus_stage2_shapes(model, t, h)
        if kernel and kernel.startswith("opus_moe2_") and "atomic" in kernel
        else None
    )
    if opus is not None:
        return _launch(
            "aiter::_opus_moe_stage2_a8w4_decode_fwd_raw", "moe2", opus, layer
        )
    return _launch(
        "pseudo_op::moe_flydsl_stage2",
        "moe2",
        scaled if scaled else ((routed, inter), (inter, h)),
        layer,
    )


def _hash_topk_shapes(model: GraphModel, tokens: int, experts: int) -> tuple:
    """Router logits, the token ids, and the ``tid2eid`` table ``(vocab, topk)``."""
    return (
        (tokens, experts),
        (tokens,),
        (model.vocab_size, model.num_experts_per_tok),
    )


def _topk_shapes(model: GraphModel, tokens: int) -> tuple:
    """Gating ids, their weights, the scores, and the per-expert bias."""
    return (
        (tokens, model.num_experts_per_tok),
        (tokens, model.num_experts_per_tok),
        (tokens, model.n_routed_experts),
        (model.n_routed_experts,),
    )


# The indexer's logits kernel schedules the keys in chunks of this many tokens
# (``block_k`` in ``aiter_fp4_paged_mqa_logits``, ``(max_seq_len + 255) // 256``
# in ``_decode_cta_count``).
_INDEXER_CHUNK_TOKENS = 256
_INDEXER_RATIO = 4


@dataclass(frozen=True)
class _Pools:
    """Sizes of the KV and state pools the run allocated, and the widths of the
    index tables built over them.

    The run's own numbers are the trace's pool facts: the length of the
    allocator's full-to-SWA map, and the request-to-token table. Everything
    else follows the engine's allocation code the way it applies them.
    """

    model: GraphModel
    slots: int
    context: int
    page: int
    blocks: int

    def _fact(self, name: str):
        return self.model.fact(name)

    @property
    def swa_rows(self) -> int:
        """The sliding-window ring, one per request slot."""
        return self.slots * self.model.sliding_window

    def kv_rows(self, ratio: int) -> int:
        """Rows of a layer's unified KV pool: the ring, the compressed
        blocks of its ratio, and one page of padding for the null slot."""
        per_block = self._fact("kv_blocks_per_c128")[ratio]
        return self.swa_rows + self.blocks * per_block + self.page // ratio

    @property
    def index_pages(self) -> int:
        """Pages of the FP4 indexer's key cache, ``(size + page + 1) // page``
        over the ratio-4 logical size at the indexer's page size."""
        per_block = self._fact("kv_blocks_per_c128")[_INDEXER_RATIO]
        page = self.page // _INDEXER_RATIO
        return (self.blocks * per_block + page + 1) // page

    def state_rows(self, ratio: int) -> int:
        """Rows of a compress state pool: the request rings, the ring again
        and a sentinel, rounded up to a multiple of the ratio."""
        ring = self._fact(f"compress_state_ring_{ratio}")
        rows = self.slots * ring + ring + 1
        return math.ceil(rows / ratio) * ratio

    @property
    def page_table_width(self) -> int:
        return math.ceil(self.context / self.page)

    def _aligned(self, width: int) -> int:
        align = self._fact("page_index_align")
        return math.ceil(width / align) * align

    @property
    def sparse_width(self) -> int:
        """Ratio-4 sparse page indices: the top-k, padded."""
        return self._aligned(self.model.index_topk)

    @property
    def dense_width(self) -> int:
        """Ratio-128 page indices: every compressed block of a full context."""
        return self._aligned(self.page_table_width * self.page // 128)

    def gathered(self, ratio: int) -> int:
        """Keys one query reads: the window and the compressed span."""
        span = self.sparse_width if ratio == 4 else self.dense_width
        return self.model.sliding_window + span

    @property
    def guarded_pages(self) -> int:
        bucket = self._fact("indexer_page_bucket")
        return max(4, math.ceil(self.page_table_width / bucket) * bucket)

    @property
    def logits_width(self) -> int:
        return self.guarded_pages * self._fact("indexer_kv_block")

    @property
    def guarded_table_width(self) -> int:
        """The guarded page table adds one scheduling chunk of look-ahead."""
        return self.guarded_pages + _INDEXER_CHUNK_TOKENS // self._fact(
            "indexer_kv_block"
        )

    def cta_count(self, tokens: int) -> int:
        chunks = max(1, math.ceil(self.logits_width / _INDEXER_CHUNK_TOKENS))
        available = tokens * chunks
        target = max(
            self._fact("indexer_cta_target"),
            tokens * self._fact("indexer_ctas_per_query"),
        )
        return min(available, target)


_POOL_FACTS = (
    "kv_mapping_rows",
    "req_to_token_shape",
    "page_size",
    "kv_blocks_per_c128",
    "compress_state_ring_4",
    "compress_state_ring_128",
    "page_index_align",
    "indexer_kv_block",
    "indexer_page_bucket",
    "indexer_cta_target",
    "indexer_ctas_per_query",
    "indexer_cta_info_width",
    "topk_plan_ints",
)


def _pools(model: GraphModel) -> _Pools | None:
    """The run's pools, or ``None`` when a fact they need is unknown."""
    if not model.sliding_window or any(
        model.fact(name) is None for name in _POOL_FACTS
    ):
        return None
    slots, context = model.fact("req_to_token_shape")
    page = model.fact("page_size")
    # The allocator's map holds ``size + page + 1`` entries. The pool configurator
    # sizes the ratio-128 cache at one block per 128 tokens of ``size``.
    size = model.fact("kv_mapping_rows") - 1 - page
    return _Pools(model, slots, context, page, size // max(model.compress_ratios))


def _mla_splits(model: GraphModel, tokens: int) -> int:
    """KV splits aiter's ``get_meta_param`` picks for the v4 decode kernel.

    The kernel passes ``ignore_total_kv=1`` and ``tg_factor`` heads groups of 64,
    so the pick is the split count (1 to 16) that fills the compute units best,
    the smallest on a tie.
    """
    if not model.num_cus:
        return 0
    occupancy = tokens * max(
        1, math.ceil(model.num_attention_heads // model.degree("attn_tp") / 64)
    )
    fill = [
        (
            occupancy
            * split
            / (
                (occupancy * split + model.num_cus - 1) // model.num_cus * model.num_cus
            ),
            split,
        )
        for split in range(1, 17)
    ]
    return max(fill, key=lambda item: item[0])[1]


def _compress_state_shapes(
    model: GraphModel, pools: _Pools, tokens: int, ratio: int, head_dim: int
) -> tuple:
    """``flash_c{4,128}_decode``: the state pool, the projected rows, the
    position bias, and the indexer's output rows when it has one."""
    width = _compressor_width(ratio, head_dim)
    coff = width // (2 * head_dim)
    return (
        (pools.state_rows(ratio) // ratio, ratio, width),
        (tokens, width),
        (coff * ratio, head_dim),
    )


def _rotary_table(model: GraphModel) -> tuple:
    rope = model.qk_rope_head_dim or model.head_dim
    return (model.max_position_embeddings, rope // 2)


def _mhc_post_shapes(model: GraphModel, tokens: int, hidden: int) -> tuple:
    """``mhc_post(out, x, residual, post_mix, comb_mix)``.

    ``out`` and ``residual`` are ``(M, n, C)`` and ``x`` is ``(M, C)``. With
    one stream the residual is just the hidden state.
    """
    streams = model.hc_mult
    if streams <= 1:
        return ((tokens, hidden),)
    return (
        (tokens, streams, hidden),
        (tokens, hidden),
        (tokens, streams, hidden),
        (tokens, streams),
        (tokens, streams, streams),
    )


def mhc_pre_split(m: int, hc_hidden: int, num_cu: int, gfx: str) -> tuple[int, int]:
    """``get_mhc_pre_splitk``: the ``(split_k, tile_k)`` the pre GEMM launches with.

    The search keeps the split whose grid fills the device, among the splits
    that divide ``hc_hidden`` and leave a full prefetch in K.
    """
    prefetch_stages = 2
    tile_m = 16 * 4
    tile_k_tg = (
        {128: 2 * num_cu, 64: 4 * num_cu}
        if gfx.startswith("gfx9")
        else {64: 4 * num_cu}
    )
    selected_splitk = 1
    selected_tile_k = 64
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, meanwhile_tg in tile_k_tg.items():
        if hc_hidden % tile_k != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hc_hidden % (splitk * tile_k) != 0 or (hc_hidden // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            score = (num_tg / meanwhile_tg) / math.ceil(num_tg / meanwhile_tg)
            if selected_score < score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            if num_tg > meanwhile_tg * 2:
                break
    return selected_splitk, selected_tile_k


def _fused_valid_splitk(hidden: int, tile_k: int, num_cu: int) -> list[int]:
    prefetch_stages = 2
    return [
        splitk
        for splitk in range(1, num_cu + 1)
        if hidden % (splitk * tile_k) == 0
        and (hidden // splitk) >= tile_k * prefetch_stages
    ]


def _fused_fill_splitk(m: int, valid: list[int], num_cu: int) -> int:
    ideal = max(1.0, 32.0 * num_cu / m)
    return min(
        valid, key=lambda splitk: (abs(math.log(splitk) - math.log(ideal)), -splitk)
    )


def _fused_config_gfx950_256(m: int, hidden: int, num_cu: int) -> int:
    tile_k = 64 if m >= 2 * hidden else 32
    if hidden % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64
    valid = _fused_valid_splitk(hidden, tile_k, num_cu)
    if not valid:
        return 1
    splitk = _fused_fill_splitk(m, valid, num_cu)
    if tile_k == 32:
        if 2 <= splitk <= 4 and (2 * splitk) in valid:
            splitk = 2 * splitk
    elif splitk < 2 and 2 in valid:
        splitk = 2
    return splitk


def _fused_config_default(m: int, hidden: int, num_cu: int) -> int:
    """The occupancy search used for a chip with no tuned fused policy."""
    prefetch_stages = 2
    tile_m = 16
    tile_k_tg = {64: 2 * num_cu, 32: 4 * num_cu}
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_splitk = 1
    selected_score = -1.0
    for tile_k, meanwhile_tg in tile_k_tg.items():
        if hidden % tile_k != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hidden % (splitk * tile_k) != 0 or (hidden // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            score = (num_tg / meanwhile_tg) / math.ceil(num_tg / meanwhile_tg)
            if score > selected_score:
                selected_splitk = splitk
                selected_score = score
            if num_tg > meanwhile_tg * 2:
                break
    return selected_splitk


def _fused_config_gfx942_80(m: int, hidden: int, num_cu: int) -> int:
    tile_k = 32 if hidden <= 4096 and m <= 128 else 64
    if hidden % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64
    valid = _fused_valid_splitk(hidden, tile_k, num_cu)
    if not valid:
        return 1
    if tile_k == 64:
        m_blocks = (m + 15) // 16
        ideal = max(1.0, 12.8 * num_cu / m_blocks)
        splitk = min(valid, key=lambda sk: (abs(math.log(sk) - math.log(ideal)), -sk))
        if splitk < 2 and 2 in valid:
            splitk = 2
        return splitk
    return _fused_fill_splitk(m, valid, num_cu)


def _fused_config_gfx1250_256(m: int, hidden: int, num_cu: int) -> int:
    tile_k = 32 if hidden % 32 == 0 else 64
    valid = _fused_valid_splitk(hidden, tile_k, num_cu)
    if not valid:
        return 1
    table = (
        (
            (128, 56),
            (256, 56),
            (512, 32),
            (1024, 8),
            (2048, 16),
            (4096, 8),
            (8192, 4),
            (1 << 30, 2),
        )
        if hidden >= 7168
        else (
            (256, 32),
            (512, 64),
            (1024, 16),
            (2048, 16),
            (4096, 16),
            (8192, 4),
            (1 << 30, 2),
        )
    )
    target = next(splitk for upper, splitk in table if m <= upper)
    return min(valid, key=lambda sk: (abs(math.log(sk) - math.log(target)), -sk))


_FUSED_SPLIT = {
    ("gfx950", 256): _fused_config_gfx950_256,
    ("gfx942", 80): _fused_config_gfx942_80,
    ("gfx1250", 256): _fused_config_gfx1250_256,
}


def mhc_fused_split(m: int, hidden: int, num_cu: int, gfx: str) -> int:
    """``get_mhc_fused_post_pre_config``'s split-K for this device."""
    policy = _FUSED_SPLIT.get((gfx, num_cu), _fused_config_default)
    return policy(m, hidden, num_cu)


def _hc3(streams: int) -> int:
    return streams * 2 + streams * streams


def _mhc_argument_shapes(
    model: GraphModel, tokens: int, hidden: int, kind: str
) -> tuple:
    """Argument shapes of one mHC kernel, in the order that kernel reads them.

    With one stream, or with no device to select a split policy, the launch
    keeps the hidden state alone. ``kind`` is ``sqrsum``, ``pre``, ``rms``,
    or ``fused``.
    """
    streams = model.hc_mult
    if streams <= 1 or model.num_cus < 1:
        return ((tokens, hidden),)
    hc3 = _hc3(streams)
    hc_hidden = streams * hidden
    if kind == "fused":
        split_k = mhc_fused_split(tokens, hidden, model.num_cus, model.gfx_arch)
    else:
        split_k, _tile_k = mhc_pre_split(
            tokens, hc_hidden, model.num_cus, model.gfx_arch
        )
    gemm_out = (split_k, tokens, hc3)
    sqrsum = (split_k, tokens)
    residual = (tokens, streams, hidden)
    fn = (hc3, hc_hidden)
    # hc_scale is the kernel's three mix coefficients. hc_base is one per mix row.
    if kind == "sqrsum":
        return (gemm_out, sqrsum, residual, fn)
    mixes = ((tokens, streams, 1), (tokens, streams, streams))
    if kind == "fused":
        return (
            gemm_out,
            sqrsum,
            residual,
            (tokens, hidden),
            residual,
            (tokens, streams),
            mixes[1],
            fn,
        )
    shapes = mixes + ((tokens, hidden), gemm_out, sqrsum, (3,), (hc3,), residual)
    if kind == "rms":
        shapes += ((hidden,),)
    return shapes


def _hc_head_shapes(model: GraphModel, tokens: int, hidden: int) -> tuple:
    """``fused_hc_head(x, hc_fn, hc_scale, hc_base)``.

    ``x`` is the streamed residual. The parameters come from
    ``make_hc_head_params``: ``hc_fn`` is ``(n, n * C)``, ``hc_scale`` is a
    single scalar and ``hc_base`` holds one value per stream.
    """
    streams = model.hc_mult
    if streams <= 1:
        return ((tokens, hidden),)
    return (
        (tokens, streams, hidden),
        (streams, streams * hidden),
        (1,),
        (streams,),
    )


def schedule_deepseek_v4(model: GraphModel, invocation: Invocation) -> list[Launch]:
    """Ordered launches of one DeepSeek-V4 graph at ``invocation``.

    Shapes are the tensors the SGLang modules are constructed with, sharded
    the way those modules shard them: ``wq_a`` is replicated, ``wq_b`` and
    ``wo_b`` are column- and row-parallel on the attention group, the routed
    experts split across the expert-parallel group, and the expert
    intermediate splits across the expert tensor-parallel group. A group the
    trace did not create has degree 1.
    """
    t = replay_size(invocation.tokens, model.capture_buckets)
    h = model.hidden_size
    tp = model.degree("tp")
    attn = model.degree("attn_tp")
    q_lora = model.q_lora_rank
    heads = model.num_attention_heads
    head = model.head_dim
    local_heads = heads // attn
    q_b_out = heads * head // attn
    groups = model.o_groups
    o_lora = model.o_lora_rank
    wo_in = heads * head // groups
    local_groups = groups // attn
    wo_b_k = groups * o_lora // attn
    inter = model.moe_intermediate_size
    experts = model.n_routed_experts
    topk = model.num_experts_per_tok
    routed = t * topk
    local_vocab = model.vocab_size // tp
    index_q_out = model.index_n_heads * model.index_head_dim

    hidden = ((t, h),)
    launches: list[Launch] = [
        _launch(
            "sglang_profiler::vocab_parallel_embedding_vocab_parallel_embedding",
            "embed",
            ((t,), (local_vocab, h)),
            None,
        ),
        _launch("aiter::cross_device_reduce", "allreduce", hidden, None),
    ]
    if model.hc_mult > 1:
        # ``hidden.unsqueeze(1).repeat(1, hc_mult, 1)`` copies the broadcast
        # hidden state into the unfolded repeat view. Both sides are bf16.
        view = (t, 1, h)
        view_stride = (h, h, 1)
        expanded, dest_stride = _unfold_repeat(view, (1, model.hc_mult, 1))
        src_stride = _expand_stride(view, view_stride, expanded)
        launches.append(
            _aten(
                "copy_",
                "copy",
                expanded,
                expanded,
                (),
                types=(_BF16, _BF16, _SCALAR),
                strides=(dest_stride, src_stride, ()),
                concrete=("", "", "False"),
            )
        )
    if invocation.prefill:
        return _schedule_prefill(
            model,
            invocation,
            launches,
            t,
            h,
            q_lora,
            q_b_out,
            head,
            wo_in,
            o_lora,
            local_groups,
            wo_b_k,
            index_q_out,
            inter,
            experts,
            routed,
        )

    launches[0:0] = _step_metadata(model, t)
    for layer in range(model.num_hidden_layers):
        ratio = model.compress_ratios[layer]
        kv = _kv_span(invocation.past_len, invocation.seq_len, ratio)
        q = (t, local_heads, head)
        launches.extend(
            _attention_block(
                model,
                layer,
                t,
                h,
                q_lora,
                q_b_out,
                q,
                kv,
                wo_in,
                o_lora,
                local_groups,
                wo_b_k,
                index_q_out,
            )
        )
        launches.extend(
            _moe_block(
                model,
                layer,
                t,
                h,
                inter,
                experts,
                routed,
                last=layer == model.num_hidden_layers - 1,
            )
        )

    launches.append(
        _launch("aiter::hc_head", "hc_head", _hc_head_shapes(model, t, h), None)
    )
    launches.extend(_logits_tail(model, t, h))
    return launches


def _schedule_prefill(
    model,
    invocation,
    launches,
    t,
    h,
    q_lora,
    q_b_out,
    head,
    wo_in,
    o_lora,
    local_groups,
    wo_b_k,
    index_q_out,
    inter,
    experts,
    routed,
) -> list[Launch]:
    """Extend graph. One indexer chunk covers 2048 query tokens.

    Both captured prefills agree: 16384 tokens launch the indexer 9 times and
    2358 tokens launch it twice, which is ``ceil((tokens + 1) / 2048)``.
    """
    chunks = max(math.ceil((t + 1) / 2048), 1)
    for layer in range(model.num_hidden_layers):
        ratio = model.compress_ratios[layer]
        kv = _kv_span(invocation.past_len, invocation.seq_len, ratio)
        q = (t, model.num_attention_heads // model.degree("attn_tp"), head)
        launches.extend(
            _prefill_attention(
                model,
                layer,
                t,
                h,
                q_lora,
                q_b_out,
                q,
                kv,
                head,
                wo_in,
                o_lora,
                local_groups,
                wo_b_k,
                index_q_out,
                chunks,
            )
        )
        launches.extend(_prefill_moe(model, layer, t, h, inter, experts, routed))
    launches.append(
        _launch("aiter::hc_head", "hc_head", _hc_head_shapes(model, t, h), None)
    )
    launches.append(_final_norm(t, h))
    # The graph hands back its two outputs, the normed hidden states and the
    # flattened streams ahead of ``hc_head``, through the shared output buffer.
    for width in (h, model.hc_mult * h):
        launches.append(_copy((t, width), _BF16, _BF16))
    return launches


def _prefill_attention(
    model,
    layer,
    t,
    h,
    q_lora,
    q_b_out,
    q,
    kv,
    head,
    wo_in,
    o_lora,
    local_groups,
    wo_b_k,
    index_q_out,
    chunks,
) -> list[Launch]:
    ratio = model.compress_ratios[layer]
    block: list[Launch] = [
        _launch(
            "aiter::mhc_pre_gemm_sqrsum",
            "mhc_sqrsum",
            _mhc_argument_shapes(model, t, h, "sqrsum"),
            layer,
        )
    ]
    if layer == 0:
        block.append(
            _launch(
                "aiter::mhc_pre_big_fuse",
                "mhc_pre",
                _mhc_argument_shapes(model, t, h, "pre"),
                layer,
            )
        )
        block.append(
            _launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, h),), layer)
        )
    else:
        block.append(
            _launch(
                "aiter::mhc_pre_big_fuse_rmsnorm",
                "mhc_pre_rms",
                _mhc_argument_shapes(model, t, h, "rms"),
                layer,
            )
        )
        block.append(
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "quant",
                _quant_shapes(model, t, h),
                layer,
            )
        )
    # Prefill runs wq_a as one CK tile GEMM, not the decode split-K pair.
    block.append(
        _launch(
            (
                "aiter::gemm_a8w8_blockscale_bpreshuffle"
                if model.weight_block_size
                else "aten::mm"
            ),
            "ck_tile",
            (
                _aiter_weight(t, h, q_lora, model.weight_block_size)
                if model.weight_block_size
                else _aten_weight(t, h, q_lora)
            ),
            layer,
        )
    )
    block.append(
        _launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, q_lora),), layer)
    )
    block.append(_ck_gemm(model, t, q_lora, q_b_out, layer))
    block.append(_launch("aiter::fuse_qk_norm_rope", "qk_rope", (q,), layer))
    if ratio == 4:
        block.extend(_prefill_c4(model, layer, t, h, q_lora, index_q_out, kv, chunks))
    else:
        block.append(_launch("aten::mm", "tensile", ((t, h), (h, head)), layer))
        block.append(_launch("sglang::flash_c128", "c128", (q, (kv, 1, head)), layer))
        block.append(_launch("sglang::write_c128", "w128", ((kv, head),), layer))
    block.extend(
        [
            _launch(
                "sglang::fused_norm_rope_flashmla",
                "flashmla",
                (q, (kv, 1, head)),
                layer,
            ),
            _launch("sglang::prefill_lengths", "pf_len", ((t,),), layer),
            *_indptr(t),
            *_indptr(t),
            _launch("sglang::prefill_indices", "pf_idx", ((t,),), layer),
            _launch(
                "pseudo_mla_prefill_fwd",
                "mla",
                (q, (kv, 1, head)),
                layer,
                (_FP8, _FP8),
            ),
            _launch("sglang::swa_scatter", "swa", (q,), layer),
            _launch("sglang::swa_scatter", "swa", (q,), layer),
            _launch(
                "sglang::apply_rotary_emb",
                "rope",
                _rope_shapes(
                    model,
                    t,
                    model.num_attention_heads // model.degree("attn_tp"),
                ),
                layer,
            ),
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "wo_quant",
                ((t, local_groups, wo_in),),
                layer,
            ),
            _launch(
                "aiter::_opus_bmm_a8w8_mxscale_raw",
                "gemm_sf",
                _wo_a_shapes(model, t, local_groups, wo_in, o_lora),
                layer,
            ),
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "quant",
                _quant_shapes(model, t, wo_b_k),
                layer,
            ),
            _ck_gemm(model, t, wo_b_k, h, layer),
            _launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer),
            _launch(
                "aiter::mhc_post", "mhc_post", _mhc_post_shapes(model, t, h), layer
            ),
        ]
    )
    return block


def _prefill_c4(model, layer, t, h, q_lora, index_q_out, kv, chunks) -> list[Launch]:
    index_dim = model.index_head_dim
    block = [
        _launch(
            "aiter::gemm_a16w16",
            "hgemm",
            _aiter_weight(t, q_lora, index_q_out, ()),
            layer,
        ),
        _launch(
            "aiter::dynamic_per_group_scaled_quant",
            "quant",
            _quant_shapes(model, t, h),
            layer,
        ),
        _ck_gemm(model, t, h, model.index_n_heads, layer),
        _launch(
            "aiter::rope_hadamard_rotate_activation_fp4quant",
            "rope_had",
            ((t, model.index_n_heads, index_dim),),
            layer,
        ),
        _launch("aten::mm", "tensile", ((t, h), (h, index_dim)), layer),
        _launch("sglang::flash_c4", "c4", ((t, index_dim), (kv, 1, index_dim)), layer),
        _launch("sglang::write_c4", "w4", ((kv, index_dim),), layer),
        _launch(
            "aiter::norm_rope_hadamard_rotate_activation_fp4quant",
            "norm_rope_had",
            ((t, model.index_n_heads, index_dim),),
            layer,
        ),
    ]
    for _chunk in range(chunks):
        block.extend(
            [
                _launch("sglang::prefill_schedule_prep", "pf_prep", ((t,),), layer),
                _launch("sglang::prefill_cta_info", "pf_cta", ((t,),), layer),
                _launch(
                    "sglang::pa_mqa_logits",
                    "indexer",
                    ((t, model.index_n_heads, index_dim), (kv, 1, index_dim)),
                    layer,
                ),
                _launch("sglang::topk", "topk_main", ((t, kv),), layer),
            ]
        )
    block.extend(
        [
            _launch("aten::mm", "tensile", ((t, h), (h, index_dim)), layer),
            _launch(
                "sglang::flash_c4", "c4", ((t, index_dim), (kv, 1, index_dim)), layer
            ),
            _launch("sglang::write_c4", "w4", ((kv, index_dim),), layer),
        ]
    )
    return block


def _prefill_moe(model, layer, t, h, inter, experts, routed) -> list[Launch]:
    block = [
        _launch(
            "aiter::mhc_pre_gemm_sqrsum",
            "mhc_sqrsum",
            _mhc_argument_shapes(model, t, h, "sqrsum"),
            layer,
        ),
        _launch(
            "aiter::mhc_pre_big_fuse_rmsnorm",
            "mhc_pre_rms",
            _mhc_argument_shapes(model, t, h, "rms"),
            layer,
        ),
        _launch("aten::mm", "gate", ((t, h), (h, experts)), layer),
    ]
    if layer < model.num_hash_layers:
        block.append(
            _launch(
                "sglang::hash_topk",
                "hash_topk",
                _hash_topk_shapes(model, t, experts),
                layer,
            )
        )
    else:
        block.append(
            _launch("aiter::topk_gating_fwd", "topk", _topk_shapes(model, t), layer)
        )
        block.append(
            _launch(
                "sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts",
                "shared",
                _topk_shapes(model, t)[:3],
                layer,
            )
        )
    packed = _moe_signature(model, t, h)
    scaled = _moe_signature(model, t, h, scales=True) if packed else None
    local_inter = (
        model.moe_intermediate_size // model.degree("moe_tp") if packed else inter
    )
    for _phase in range(4):
        block.append(
            _launch("aiter::moe_sorting_opus_fwd", "opus_sort", ((t, experts),), layer)
        )
    block.extend(
        [
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "quant",
                _quant_shapes(model, t if packed else routed, h),
                layer,
            ),
            _launch(
                "aiter::mxfp4_moe_sort",
                "mxfp4",
                ((t, h),) if packed else ((routed, h),),
                layer,
            ),
            _launch(
                "pseudo_op::moe_flydsl_stage1",
                "moe1",
                scaled if packed else ((routed, h), (h, 2 * inter)),
                layer,
            ),
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "quant",
                _quant_shapes(
                    model, t if packed else routed, local_inter if packed else inter
                ),
                layer,
            ),
            _launch(
                "aiter::mxfp4_moe_sort",
                "mxfp4",
                ((t, local_inter),) if packed else ((routed, inter),),
                layer,
            ),
            _stage2_launch(model, t, h, scaled, layer, routed, inter),
        ]
    )
    if packed and _stage2_reduces(model, t):
        block.append(
            _launch(
                "pseudo_op::moe_flydsl_stage2",
                "moe_red",
                scaled,
                layer,
                same_call=True,
            )
        )
    block.extend(
        [
            _launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer),
            _launch(
                "aiter::mhc_post", "mhc_post", _mhc_post_shapes(model, t, h), layer
            ),
        ]
    )
    return block


def _qk_rope_shapes(model, pools: _Pools | None, t: int, q: tuple, ratio: int):
    """``fuse_qk_norm_rope``: the query, the key row and its rotary slice, the
    norm weight, the outputs, positions, both rotary tables, and the unified
    cache the key is written to with its rotary half."""
    if pools is None:
        return (q,)
    head = model.head_dim
    rope = model.qk_rope_head_dim or head
    rows = pools.kv_rows(ratio)
    table = (model.max_position_embeddings, 1, 1, rope // 2)
    return (
        q,
        (t, head),
        (t, 1, rope),
        (head,),
        (t, 1, head),
        q,
        (t,),
        table,
        table,
        (t, q[1], rope),
        (rows, head),
        (rows, rope),
        (t,),
        (t,),
    )


def _flash_compress_shapes(model, pools, t, q, kv, ratio, head):
    """``flash_c{4,128}_decode`` on the attention compressor's state."""
    if pools is None:
        if ratio == 4:
            return ((t, head), (kv, 1, head))
        return (q, (kv, 1, head))
    return _compress_state_shapes(model, pools, t, ratio, head)


def _flashmla_store_shapes(model, pools, t, q, kv, ratio):
    """``fused_norm_rope_flashmla``: the compressed row, its norm weight, the
    rotary table, positions, and the unified cache with its rotary bytes."""
    if pools is None:
        return (q, (kv, 1, model.head_dim))
    head = model.head_dim
    rope = model.qk_rope_head_dim or head
    rows = pools.kv_rows(ratio)
    # The rotary half is stored in bf16, two bytes a value.
    return (
        (t, head),
        (head,),
        _rotary_table(model),
        (t,),
        (rows, head),
        (rows, 2 * rope),
    )


def _mla_shapes(model, pools, t, q, kv, ratio):
    """``mla_decode_fwd_v4_nm``: query and rotary query, the cache and its
    rotary view at page size one, the indptr and index arrays, the sink, and
    the split outputs."""
    if pools is None:
        return (q, (kv, 1, model.head_dim))
    head = model.head_dim
    rope = model.qk_rope_head_dim or head
    rows = pools.kv_rows(ratio)
    heads = q[1]
    splits = _mla_splits(model, t)
    # The four scalars sit between the sink and the split buffers, and the
    # valid-split flag plus the unused last-page tensor follow the output.
    split_out = ((t, splits, heads, head), (t, splits, heads, 1)) if splits else ()
    scalars = ((), (), (), ()) if splits else ()
    tail = ((), ()) if splits else ()
    return (
        q,
        (t, heads, rope),
        (rows, 1, 1, head),
        (rows, 1, 1, rope),
        (t + 1,),
        (t + 1,),
        (t * pools.gathered(ratio),),
        (t + 1,),
        (heads,),
        *scalars,
        *split_out,
        q,
        (t,),
        *tail,
    )


def _mla_concrete(splits: int) -> tuple[str, ...]:
    """Scalar values ``mla_decode_v4_asm`` is called with on the decode path.

    ``max_seqlen_q`` is 1. The kernel ignores ``softmax_scale`` and the wrapper
    passes 0. ``out_16_nosplit`` is the bf16 path only when there is one split.
    ``use_valid_split_count_reduce`` is set when the kernel writes the per-request
    split count. Tensor arguments are blank, in the order the profiler records.
    """
    if splits <= 0:
        return ()
    nosplit = splits == 1
    values = [""] * 19
    values[9] = "1"
    values[10] = "0."
    values[11] = "1" if nosplit else "0"
    values[12] = str(splits)
    values[17] = "0" if nosplit else "1"
    return tuple(values)


def _stage2_shapes(model, pools, t, q, kv, ratio):
    """``decode_fp8_2buff``: query, rotary query, the two cache buffers, the
    gathered indices, their indptr, and the sink."""
    if pools is None:
        return (q, (kv, 1, model.head_dim))
    head = model.head_dim
    rope = model.qk_rope_head_dim or head
    rows = pools.kv_rows(ratio)
    heads = q[1]
    return (
        q,
        (t, heads, rope),
        (rows, head),
        (rows, rope),
        (t * pools.gathered(ratio),),
        (t + 1,),
        (heads,),
    )


def _fill_tail_shapes(model, pools: _Pools, tokens: int, ratio: int) -> tuple:
    """``_fill_compress_tail_kernel(indices, indptr, lengths, tail, ...)``: the
    gathered page indices of the batch, and the tail each row's compressed span
    pads to."""
    width = pools.sparse_width if ratio == 4 else pools.dense_width
    return (
        (tokens * pools.gathered(ratio),),
        (tokens + 1,),
        (tokens,),
        (tokens, width),
        (tokens,),
    )


def _aten(
    name: str,
    key: str,
    *shapes: tuple[int, ...],
    types: tuple[str, ...] = (),
    strides: tuple[tuple[int, ...], ...] = (),
    concrete: tuple[str, ...] = (),
) -> Launch:
    """A torch op the replay launches eagerly; ``()`` marks a scalar operand."""
    return _launch(
        f"aten::{name}",
        key,
        shapes,
        None,
        types or None,
        concrete,
        strides=strides,
    )


def _copy(
    shape: tuple[int, ...],
    dst: str,
    src: str,
    dst_stride: tuple[int, ...] | None = None,
    src_stride: tuple[int, ...] | None = None,
) -> Launch:
    """``aten::copy_`` of one tensor into another. The third argument is
    ``non_blocking``, a scalar the copy kernel does not read."""
    packed = _row_major(shape)
    return _aten(
        "copy_",
        "copy",
        shape,
        shape,
        (),
        types=(dst, src, _SCALAR),
        strides=(dst_stride or packed, src_stride or packed, ()),
        concrete=("", "", "False"),
    )


def _scalar_binop(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    stride: tuple[int, ...] | None = None,
) -> Launch:
    """``tensor + python int`` or ``tensor - python int``.

    The Python int is recorded as a long scalar and the alpha is 1, so the
    kernel is the functor of the tensor's dtype.
    """
    return _aten(
        name,
        "t_add",
        shape,
        (),
        (),
        types=(dtype, _I64, _SCALAR),
        strides=(stride or _row_major(shape), (), ()),
        concrete=("", "", "1"),
    )


def _indptr(t: int) -> list[Launch]:
    """``_lengths_to_indptr``: ``F.pad(cumsum(lengths, dtype=int32), (1, 0))``.

    The cumsum is int32, so the pad's copy is an int32 memcpy.
    """
    return [
        _aten("cumsum", "t_scan", (t,), (), ()),
        _aten("fill_", "t_fill", (t + 1,), ()),
        _copy((t,), _I32, _I32),
    ]


def _request_broadcast_stride(tokens: int, window: int) -> tuple[int, int]:
    """``req_pool_indices`` broadcast across the sliding window.

    Several requests are a ``(tokens, 1)`` column repeated over the window.
    One request is a length-1 vector, and expanding that vector over the
    window has a zero step on both axes.
    """
    if tokens == 1:
        return _expand_stride((tokens,), (1,), (tokens, window))
    return _expand_stride((tokens, 1), (1, 1), (tokens, window))


def _swa_window_indices(model: GraphModel, pools: _Pools, t: int) -> list[Launch]:
    """``make_core_attn_metadata``: the sliding window's page indices, built from
    the positions, the request-to-token table and the full-to-SWA map.

    ``seq_lens`` is int64 and the backend casts it to int32. The window
    subtraction is that int32 vector minus ``arange(window)``. Indexing casts
    each index to int64: the int32 offsets, then the request slot broadcast
    across the window, then the int32 locations read from ``req_to_token``
    before the int64 full-to-SWA map. ``flash_mla`` wants int32 indices, and
    the pad of an already-aligned window copies those int32 values.
    """
    w = model.sliding_window
    table = (pools.slots, pools.context)
    window = (t, w)
    return [
        _copy((t,), _I32, _I64),
        _scalar_binop("sub", (t,), _I32),
        _aten("arange", "t_arange", (), (), (), (0,)),
        _aten(
            "sub",
            "t_add",
            (t, 1),
            (1, w),
            (),
            types=(_I32, _I32, _SCALAR),
            strides=((1, 1), (w, 1), ()),
            concrete=("", "", "1"),
        ),
        _aten("lt", "t_cmp", (t, w), ()),
        _aten("masked_fill_", "t_mfill", (t, w), (t, w), ()),
        _copy(window, _I64, _I32),
        _copy(window, _I64, _I64, src_stride=_request_broadcast_stride(t, w)),
        _aten("index", "t_index", table, ()),
        _aten("masked_fill_", "t_mfill", (t, w), (t, w), ()),
        _copy(window, _I64, _I32),
        _aten("index", "t_index", (model.fact("kv_mapping_rows"),), ()),
        _copy(window, _I32, _I64),
        _copy(window, _I32, _I32),
        _scalar_binop("sub", (t,), _I32),
        _aten("clamp", "t_clamp", (t,), (), ()),
        _aten("index", "t_index", (pools.slots, pools.page_table_width), ()),
        _aten("floor_divide", "t_fdiv", (t, pools.page_table_width), ()),
    ]


def _compression_indices(model: GraphModel, pools: _Pools, t: int) -> list[Launch]:
    """``init_compression_metadata`` and ``init_flashmla_related``: the ratio-128
    page indices padded to the index alignment and the ratio-4 top-k clamps."""
    w = model.sliding_window
    launches = [
        _launch(
            "sglang::_init_compressed_attn_metadata",
            "init_meta",
            ((t,), (t,), (t,), (t, pools.page_table_width)),
            None,
        )
    ]
    if 128 in model.compress_ratios:
        # ``_pad_last_dim`` copies the raw table into the left of the aligned
        # buffer, so the destination row is the aligned width wide.
        dense = pools.page_table_width * pools.page // 128
        aligned = pools.dense_width
        launches += [
            _aten("fill_", "t_fill", (t, aligned), ()),
            _copy(
                (t, dense),
                _I32,
                _I32,
                dst_stride=(aligned, 1),
                src_stride=(dense, 1),
            ),
            _copy(
                (t, w),
                _I32,
                _I32,
                dst_stride=(pools._aligned(w), 1),
                src_stride=(w, 1),
            ),
        ]
    # The compression kernel writes ``c4_out_loc`` and ``c128_out_loc`` as
    # int64. Adding the SWA page offset keeps that dtype.
    launches += [
        _scalar_binop("add", (t,), _I64),
        _scalar_binop("add", (t,), _I64),
    ]
    if _INDEXER_RATIO in model.compress_ratios:
        top = model.index_topk
        launches += [
            _aten("clamp", "t_clamp", (t,), (), ()),
            _aten("clamp", "t_clamp", (t,), (), ()),
            _aten("fill_", "t_fill", (t, pools.sparse_width), ()),
            _copy(
                (t, top),
                _I32,
                _I32,
                dst_stride=(pools.sparse_width, 1),
                src_stride=(top, 1),
            ),
        ]
    return launches


def _decode_streams(model: GraphModel, pools: _Pools, t: int) -> list[Launch]:
    """``runtime.build_decode_streams``: the three KV streams' index pointers and
    gather indices, then the ring-write slots ``_attach_unified_kv_decode_streams``
    derives from the request slots."""
    # ``state_slot`` is the int64 request pool. The stream builder casts it
    # to int32, and the three indptr pads copy int32 cumsums.
    launches = [_copy((t,), _I32, _I64)]
    launches += _indptr(t)
    for _ in range(2):
        launches.append(
            _aten(
                "add",
                "t_add",
                (t,),
                (t,),
                (),
                types=(_I32, _I32, _SCALAR),
                strides=((1,), (1,), ()),
                concrete=("", "", "1"),
            )
        )
        launches += _indptr(t)
    launches.append(_aten("arange", "t_arange", (), (), (), (0,)))
    gathered = (
        t * model.sliding_window,
        t * pools.gathered(_INDEXER_RATIO),
        t * pools.gathered(128),
    )
    launches.append(
        _launch(
            "sglang::paged_decode_indices_write",
            "paged_idx",
            ((t,),) * 3 + ((t + 1,),) * 3 + tuple((n,) for n in gathered),
            None,
        )
    )
    if 128 in model.compress_ratios:
        launches.append(
            _launch(
                "sglang::fill_compress_tail",
                "fill_tail",
                _fill_tail_shapes(model, pools, t, 128),
                None,
            )
        )
    # ``req_slot * ring + positions.to(int64) % ring``, then back to int32.
    # The request slot is already int64, so only the positions cast copies.
    launches += [
        _aten(
            "mul",
            "t_mul",
            (t,),
            (),
            types=(_I64, _I64),
            strides=((1,), ()),
            concrete=("", ""),
        ),
        _copy((t,), _I64, _I32),
        _aten(
            "remainder",
            "t_rem",
            (t,),
            (),
            types=(_I64, _I64),
            strides=((1,), ()),
            concrete=("", ""),
        ),
        _aten(
            "add",
            "t_add",
            (t,),
            (t,),
            (),
            types=(_I64, _I64, _SCALAR),
            strides=((1,), (1,), ()),
            concrete=("", "", "1"),
        ),
        _copy((t,), _I32, _I64),
    ]
    return launches


def _indexer_write_metadata(model: GraphModel, pools: _Pools, t: int) -> list[Launch]:
    """``init_forward_metadata_in_graph``: the compressor plans, the output cache
    slots, ``prepare_fp4_k_write_metadata`` and the indexer's CTA schedule."""
    table = (pools.slots, pools.context)
    plan = _launch(
        "sglang::compressor_v2_create_paged_compressor_data",
        "plan",
        (table, (t,), (t,), ()),
        None,
    )
    launches = [plan] * sum(ratio in model.compress_ratios for ratio in (4, 128))
    # ``translate_loc_from_full_to_swa(out_cache_loc).to(int32)``. Both the
    # cache location and the map are int64.
    launches += [
        _aten("index", "t_index", (model.fact("kv_mapping_rows"),), ()),
        _copy((t,), _I32, _I64),
    ]
    if _INDEXER_RATIO not in model.compress_ratios:
        return launches
    # ``plan_words[:, 0].to(int64)`` reads a column of the int32 view of the
    # uint8 plan. A one-element column is contiguous, so the cast keeps its
    # step; ``seq_lens - ratio`` and ``seq_lens % ratio`` then read that step.
    column = (_DECODE_PLAN_BYTES // 4,)
    kept = _dtype_cast_stride((t,), column)
    launches += [
        _copy((t,), _I64, _I32, dst_stride=kept, src_stride=column),
        _scalar_binop("sub", (t,), _I64, stride=kept),
        _aten("ge", "t_cmp", (t,), ()),
        _aten("lt", "t_cmp", (t,), ()),
        _aten("bitwise_and", "t_and", (t,), (t,)),
        _aten("fill_", "t_fill", (t,), ()),
        _aten("where", "t_where", (t,), (t,), (t,)),
        _aten(
            "remainder",
            "t_rem",
            (t,),
            (),
            types=(_I64, _I64),
            strides=(kept, ()),
            concrete=("", ""),
        ),
        _aten("eq", "t_eq", (t,), ()),
        _aten("bitwise_and_", "t_and", (t,), (t,)),
        _aten("fill_", "t_fill", (t,), ()),
        _aten("where", "t_where", (t,), (t,), (t,)),
        # ``fp4_q_positions = positions.to(int64)``. positions is the int32
        # casual position, so this is a packed cast.
        _copy((t,), _I64, _I32),
        _launch(
            "sglang::pad_page_table", "pad_pt", ((t, pools.page_table_width),), None
        ),
    ]
    schedule = (
        (t,),
        (pools.cta_count(t), model.fact("indexer_cta_info_width")),
    )
    launches += [
        _launch("sglang::build_decode_schedule", "sched_prep", schedule, None),
        _launch(
            "sglang::build_decode_schedule",
            "cta_info",
            schedule,
            None,
            same_call=True,
        ),
    ]
    return launches


def _step_metadata(model: GraphModel, tokens: int) -> list[Launch]:
    """Kernels a decode replay runs before the model: the attention metadata the
    backend builds from the batch (window indices, compression indices, KV
    streams, compressor plans, FP4 write metadata, indexer schedule)."""
    pools = _pools(model)
    if pools is None:
        return []
    return (
        _swa_window_indices(model, pools, tokens)
        + _compression_indices(model, pools, tokens)
        + _decode_streams(model, pools, tokens)
        + _indexer_write_metadata(model, pools, tokens)
    )


def _final_norm(tokens: int, hidden: int) -> Launch:
    return _launch(
        "aiter::rmsnorm",
        "rmsnorm",
        ((tokens, hidden), (tokens, hidden), (hidden,)),
        None,
    )


def _logits_tail(model: GraphModel, tokens: int, hidden: int) -> list[Launch]:
    """Final norm, LM head, and the logits processor's all-gather and casts.

    The gathered ``(tp, tokens, vocab / tp)`` shards move to ``(tokens, tp,
    vocab / tp)``, a copy once there is more than one token to interleave.
    """
    local_vocab = model.vocab_size // model.degree("tp")
    launches = [
        _final_norm(tokens, hidden),
        _launch("aten::mm", "tensile", ((tokens, hidden), (hidden, local_vocab)), None),
        _launch(
            "aiter::reg_all_gather",
            "allgather",
            ((model.degree("tp") * tokens, local_vocab), (tokens, local_vocab)),
            None,
        ),
    ]
    if tokens > 1:
        # The all-gather buffer is ``(tp, tokens, vocab / tp)``, contiguous.
        # ``permute(1, 0, 2)`` moves it to ``(tokens, tp, vocab / tp)``, so the
        # source strides follow that order and the destination is packed.
        gathered = (tokens, model.degree("tp"), local_vocab)
        packed = _row_major((model.degree("tp"), tokens, local_vocab))
        src_stride = (packed[1], packed[0], packed[2])
        launches.append(_copy(gathered, _BF16, _BF16, src_stride=src_stride))
    logits = (tokens, model.vocab_size)
    launches.append(_copy(logits, "float", _BF16))
    return launches


def _attention_block(
    model,
    layer,
    t,
    h,
    q_lora,
    q_b_out,
    q,
    kv,
    wo_in,
    o_lora,
    local_groups,
    wo_b_k,
    index_q_out,
) -> list[Launch]:
    ratio = model.compress_ratios[layer]
    head = model.head_dim
    block: list[Launch] = []
    if layer == 0:
        block.append(
            _launch(
                "aiter::mhc_pre_gemm_sqrsum",
                "mhc_sqrsum",
                _mhc_argument_shapes(model, t, h, "sqrsum"),
                layer,
            )
        )
        block.append(
            _launch(
                "aiter::mhc_pre_big_fuse",
                "mhc_pre",
                _mhc_argument_shapes(model, t, h, "pre"),
                layer,
            )
        )
        block.append(
            _launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, h),), layer)
        )
    else:
        block.append(
            _launch(
                "aiter::mhc_pre_big_fuse_rmsnorm",
                "mhc_pre_rms",
                _mhc_argument_shapes(model, t, h, "rms"),
                layer,
            )
        )
        block.append(
            _launch(
                "aiter::dynamic_per_group_scaled_quant",
                "quant",
                _quant_shapes(model, t, h),
                layer,
            )
        )
    # wq_a and wkv are replicated and read the same activation. SGLang fuses
    # them into ``wqkv_a`` when ``SGLANG_OPT_FUSE_WQA_WKV`` is on, which is the
    # default. The split-K gemm launches the product and its reduce. The weight
    # is the parameter (out, in), and the scales follow the block.
    a_widths = (q_lora + head,) if model.fact("fuse_wqa_wkv", True) else (q_lora, head)
    for width in a_widths:
        wq_a = _aiter_weight(t, h, width, model.weight_block_size)
        block.append(
            _launch("aiter::gemm_a8w8_blockscale_bpreshuffle", "gemm_a8w8", wq_a, layer)
        )
        block.append(
            _launch(
                "aiter::gemm_a8w8_blockscale_bpreshuffle",
                "gemm_red",
                wq_a,
                layer,
                same_call=True,
            )
        )
    block.append(
        _launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, q_lora),), layer)
    )
    # wq_b is column-parallel on the attention group.
    block.append(_ck_gemm(model, t, q_lora, q_b_out, layer))
    pools = _pools(model)
    block.append(
        _launch(
            "aiter::fuse_qk_norm_rope",
            "qk_rope",
            _qk_rope_shapes(model, pools, t, q, ratio),
            layer,
        )
    )
    if ratio == 4:
        block.extend(_c4_indexer(model, layer, t, h, q_lora, index_q_out, kv))
    # The attention compressor projects ``wkv_gate`` on the hidden state, then
    # compresses. The window overlap of ratio 4 doubles its width.
    block.append(
        _launch(
            "aten::mm",
            "tensile",
            _aten_weight(t, h, _compressor_width(ratio, head)),
            layer,
        )
    )
    compress_key = "c4" if ratio == 4 else "c128"
    block.append(
        _launch(
            f"sglang::flash_{compress_key}",
            compress_key,
            _flash_compress_shapes(model, pools, t, q, kv, ratio, head),
            layer,
        )
    )
    block.append(
        _launch(
            "sglang::fused_norm_rope_flashmla",
            "flashmla",
            _flashmla_store_shapes(model, pools, t, q, kv, ratio),
            layer,
        )
    )
    if ratio == 4:
        # The compressed tail is written after the flashmla norm, before MLA.
        block.append(
            _launch(
                "sglang::fill_compress_tail",
                "fill_tail",
                _fill_tail_shapes(model, pools, t, 4) if pools else ((kv, head),),
                layer,
            )
        )
    mla = _mla_shapes(model, pools, t, q, kv, ratio)
    block.append(
        _launch(
            "aiter::mla_decode_v4_asm",
            "mla",
            mla,
            layer,
            concrete=_mla_concrete(_mla_splits(model, t) if pools else 0),
        )
    )
    block.append(
        _launch(
            "aiter::mla_decode_stage2",
            "attn_asm",
            _stage2_shapes(model, pools, t, q, kv, ratio),
            layer,
        )
    )
    block.append(
        _launch(
            "sglang::apply_rotary_emb",
            "rope",
            _rope_shapes(model, t, q[1] if len(q) > 1 else 1),
            layer,
        )
    )
    block.append(
        _launch(
            "aiter::dynamic_per_group_scaled_quant",
            "wo_quant",
            ((t, local_groups, wo_in),),
            layer,
        )
    )
    # wo_a is one grouped GEMM per local attention group.
    block.append(
        _launch(
            "aiter::_opus_bmm_a8w8_mxscale_raw",
            "gemm_mx",
            _wo_a_shapes(model, t, local_groups, wo_in, o_lora),
            layer,
        )
    )
    block.append(
        _launch(
            "aiter::dynamic_per_group_scaled_quant",
            "quant",
            _quant_shapes(model, t, wo_b_k),
            layer,
        )
    )
    # wo_b is row-parallel: this GPU multiplies its K slice into the full hidden.
    block.append(_ck_gemm(model, t, wo_b_k, h, layer))
    block.append(_launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer))
    block.append(
        _launch(
            "aiter::mhc_fused_post_pre_gemm_sqrsum",
            "mhc_fused",
            _mhc_argument_shapes(model, t, h, "fused"),
            layer,
        )
    )
    return block


def _c4_indexer(model, layer, t, h, q_lora, index_q_out, kv) -> list[Launch]:
    """The ratio-4 indexer, in the order ``C4Indexer`` and its backend issue it.

    ``weights_proj`` is a bf16 ``Linear(hidden, n_heads)``. ``wq_b`` reads the
    quantized ``q_lora`` and projects to ``n_heads * head_dim``. The query is
    rotated and quantized to FP4, then the indexer's own compressor projects
    ``wkv_gate`` and compresses the keys it will score against.
    """
    index_dim = model.index_head_dim
    pools = _pools(model)
    return [
        _launch(
            "aiter::gemm_a16w16",
            "hgemm",
            _aiter_weight(t, h, model.index_n_heads, ()),
            layer,
        ),
        _launch(
            "aiter::dynamic_per_group_scaled_quant",
            "quant",
            _quant_shapes(model, t, q_lora),
            layer,
        ),
        _ck_gemm(model, t, q_lora, index_q_out, layer),
        _launch(
            "aiter::rope_hadamard_rotate_activation_fp4quant",
            "rope_had",
            _indexer_rotary_shapes(model, t),
            layer,
        ),
        _launch(
            "aten::mm",
            "tensile",
            _aten_weight(t, h, _compressor_width(4, index_dim)),
            layer,
        ),
        _launch(
            "sglang::flash_c4",
            "c4",
            _indexer_compress_shapes(model, pools, t, kv),
            layer,
        ),
        _launch(
            "aiter::norm_rope_hadamard_rotate_activation_fp4quant",
            "norm_rope_had",
            _indexer_cache_write_shapes(model, pools, t),
            layer,
        ),
        _launch(
            "sglang::pa_mqa_logits",
            "indexer",
            _indexer_logits_shapes(model, pools, t, kv),
            layer,
        ),
        _launch(
            "sglang::topk",
            "topk_main",
            _topk_main_shapes(model, pools, t, kv),
            layer,
        ),
    ]


def _indexer_compress_shapes(model, pools, t, kv):
    """The indexer compressor's ``flash_c4_decode``: its state pool, the projected
    rows, the position bias, and the compressed rows it returns."""
    index_dim = model.index_head_dim
    if pools is None:
        return ((t, index_dim), (kv, 1, index_dim))
    return (
        *_compress_state_shapes(model, pools, t, _INDEXER_RATIO, index_dim),
        (t, index_dim),
    )


def _indexer_cache_write_shapes(model, pools, t):
    """``norm_rope_hadamard_rotate_activation_fp4quant_kvcache``: the FP4 key
    cache and its scales, the key row, the norm weight, both rotary tables,
    and the cache locations."""
    index_dim = model.index_head_dim
    if pools is None:
        return ((t, model.index_n_heads, index_dim),)
    table = _rotary_table(model)
    return (
        (pools.index_pages, 1, 4, pools.page // _INDEXER_RATIO, 16),
        (pools.index_pages, 1, 4, pools.page // _INDEXER_RATIO),
        (t, 1, index_dim),
        (index_dim,),
        table,
        table,
        (t,),
        (t,),
    )


def _indexer_logits_shapes(model, pools, t, kv):
    """``pa_mqa_logits_fp4``: the packed query and its scale, the key cache and
    scales, the guarded page table, the head weights, and the lengths."""
    index_dim = model.index_head_dim
    heads = model.index_n_heads
    scale = model.fact("indexer_q_scale_shape")
    if pools is None or scale is None:
        return ((t, heads, index_dim), (kv, 1, index_dim))
    block = model.fact("indexer_kv_block")
    return (
        (t, 1, heads, index_dim // 2),
        (t, 1, *scale),
        (pools.index_pages, 1, 4, block, 16),
        (pools.index_pages, 1, 4, block),
        (t, pools.guarded_table_width),
        (t, heads),
        (t,),
    )


def _topk_main_shapes(model, pools, t, kv):
    """``topk_main_kernel``: the logits, the lengths, the page table, the
    selected indices, and the plan, which holds a row per request and one more."""
    if pools is None:
        return ((t, kv),)
    return (
        (t, pools.logits_width),
        (t,),
        (t, pools.page_table_width),
        (t, model.index_topk),
        (t + 1, model.fact("topk_plan_ints")),
    )


def _moe_block(model, layer, t, h, inter, experts, routed, last: bool) -> list[Launch]:
    packed = _moe_signature(model, t, h)
    scaled = _moe_signature(model, t, h, scales=True) if packed else None
    local_inter = (
        model.moe_intermediate_size // model.degree("moe_tp") if packed else inter
    )
    block = [
        _launch(
            "aiter::mhc_pre_big_fuse_rmsnorm",
            "mhc_pre_rms",
            _mhc_argument_shapes(model, t, h, "rms"),
            layer,
        ),
        _launch(
            "aiter::gemm_a16w16",
            "hgemm",
            _aiter_weight(t, h, experts, ()),
            layer,
        ),
    ]
    if layer < model.num_hash_layers:
        block.append(
            _launch(
                "sglang::hash_topk",
                "hash_topk",
                _hash_topk_shapes(model, t, experts),
                layer,
            )
        )
    else:
        block.append(
            _launch("aiter::topk_gating_fwd", "topk", _topk_shapes(model, t), layer)
        )
        # This kernel appends the shared-expert id onto the top-k ids.
        block.append(
            _launch(
                "sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts",
                "shared",
                _topk_shapes(model, t)[:3],
                layer,
            )
        )
    block.extend(
        [
            _launch(
                "aiter::moe_sorting_fwd",
                "moe_sort",
                scaled if packed else ((t, experts),),
                layer,
            ),
            _launch(
                "aiter::fused_dynamic_mx_quant_moe_sort_hip",
                "mx_sort",
                _moe_sort_shapes(model, t, h, 1)
                or (((t, h),) if packed else ((routed, h),)),
                layer,
            ),
            _launch(
                "pseudo_op::moe_flydsl_stage1",
                "moe1",
                scaled if packed else ((routed, h), (h, 2 * inter)),
                layer,
            ),
            _launch(
                "aiter::fused_dynamic_mx_quant_moe_sort_hip",
                "mx_sort",
                _moe_sort_shapes(model, t, h, 2)
                or (((t, local_inter),) if packed else ((routed, inter),)),
                layer,
            ),
            _stage2_launch(model, t, h, scaled, layer, routed, inter),
        ]
    )
    if packed and _stage2_reduces(model, t):
        # The reduction is launched by the same stage-2 call, so it is recorded
        # with that call's tensors.
        block.append(
            _launch(
                "pseudo_op::moe_flydsl_stage2",
                "moe_red",
                scaled,
                layer,
                same_call=True,
            )
        )
    block.append(_launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer))
    if last:
        block.append(
            _launch("aiter::mhc_post", "mhc_post", _mhc_post_shapes(model, t, h), layer)
        )
    else:
        block.append(
            _launch(
                "aiter::mhc_fused_post_pre_gemm_sqrsum",
                "mhc_fused",
                _mhc_argument_shapes(model, t, h, "fused"),
                layer,
            )
        )
    return block


def launches_from_architecture(
    spec, model: GraphModel, invocation: Invocation
) -> list[Launch] | None:
    """Launches read from the engine modeling source, when that source states shapes.

    The exporter loads this spec from the pinned checkout of the engine that
    produced the trace. A node whose details name a kernel and the tensors it
    is called with becomes one launch. A spec that does not state those
    tensors leaves the config schedule in place.
    """
    trees = getattr(spec, "export_block_trees", None) or []
    found: list[Launch] = []
    for _title, root in trees:
        _collect_kernel_launches(root, found)
    if not found:
        return None
    tokens = replay_size(invocation.tokens, model.capture_buckets)
    return [_with_token_count(launch, tokens) for launch in found]


def _collect_kernel_launches(node, found: list[Launch]) -> None:
    details = [str(line) for line in (getattr(node, "details", None) or [])]
    kernel = next(
        (
            line.split(":", 1)[1].strip()
            for line in details
            if line.startswith("kernel:")
        ),
        None,
    )
    shapes = _shapes_in_details(details)
    if kernel and shapes:
        found.append(Launch(kernel, kernel, shapes))
    for child in getattr(node, "children", None) or []:
        _collect_kernel_launches(child, found)


def _shapes_in_details(details: list[str]) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []
    for line in details:
        if not line.startswith("shape:"):
            continue
        raw = line.split(":", 1)[1].strip()
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if (
            isinstance(parsed, tuple)
            and parsed
            and all(isinstance(item, tuple) for item in parsed)
            and all(all(isinstance(dim, int) for dim in item) for item in parsed)
        ):
            shapes.extend(parsed)
        elif isinstance(parsed, tuple) and all(isinstance(dim, int) for dim in parsed):
            shapes.append(parsed)
    return tuple(shapes)


def _with_token_count(launch: Launch, tokens: int) -> Launch:
    """Replace a leading 0, the unset token axis, with this invocation's width."""
    shapes = tuple(
        ((tokens,) + shape[1:]) if shape and shape[0] == 0 else shape
        for shape in launch.input_shapes
    )
    return replace(launch, input_shapes=shapes)


def covering_annotation(events: list[dict], launch: dict) -> str | None:
    """The step annotation whose time span contains ``launch``, if there is one.

    The tightest containing annotation wins, so a step inside a wider range
    is the invocation this graph belongs to.
    """
    start = launch.get("ts")
    if start is None:
        return None
    end = start + (launch.get("dur") or 0)
    best_name = None
    best_span = None
    for event in events:
        name = event.get("name") or ""
        if "step[" not in name and "bs=" not in name:
            continue
        if not parse_annotation(name).matched:
            continue
        ev_start = event.get("ts")
        if ev_start is None:
            continue
        ev_end = ev_start + (event.get("dur") or 0)
        if ev_start <= start and end <= ev_end:
            span = ev_end - ev_start
            if best_span is None or span < best_span:
                best_span = span
                best_name = name
    return best_name
