###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Launch list of a DeepSeek-V4 graph, with the input shapes of each launch.

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

import math
from dataclasses import dataclass

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


def graph_model_from_config(config: dict, tp: int) -> GraphModel:
    """Build a :class:`GraphModel` from a Hugging Face ``config.json``."""
    ratios = config.get("compress_ratios") or ()
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
    )


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
    return assigned


def _launch(
    op_name: str,
    key: str,
    shapes: tuple[tuple[int, ...], ...],
    layer: int | None,
) -> Launch:
    return Launch(op_name, key, shapes, layer)


def schedule_deepseek_v4(model: GraphModel, invocation: Invocation) -> list[Launch]:
    """Ordered launches of one DeepSeek-V4 graph at ``invocation``.

    Shapes are the tensors the SGLang modules are constructed with, sharded by
    ``model.tp`` the way those modules shard them: ``wq_a`` is replicated,
    ``wq_b`` and ``wo_b`` are column- and row-parallel on the attention group,
    and the routed experts stay on this GPU (expert parallel degree 1).
    """
    t = invocation.tokens
    h = model.hidden_size
    tp = model.tp
    q_lora = model.q_lora_rank
    heads = model.num_attention_heads
    head = model.head_dim
    local_heads = heads // tp
    q_b_out = heads * head // tp
    groups = model.o_groups
    o_lora = model.o_lora_rank
    wo_in = heads * head // groups
    local_groups = groups // tp
    wo_b_k = groups * o_lora // tp
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
    if invocation.prefill:
        return _schedule_prefill(
            model, invocation, launches, t, h, q_lora, q_b_out, head,
            wo_in, o_lora, local_groups, wo_b_k, index_q_out, inter, experts, routed,
        )

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

    launches.append(_launch("aiter::hc_head", "hc_head", hidden, None))
    launches.append(
        _launch(
            "aten::mm",
            "tensile",
            ((t, h), (h, local_vocab)),
            None,
        )
    )
    return launches


def _schedule_prefill(
    model, invocation, launches, t, h, q_lora, q_b_out, head,
    wo_in, o_lora, local_groups, wo_b_k, index_q_out, inter, experts, routed,
) -> list[Launch]:
    """Extend graph. One indexer chunk covers 2048 query tokens.

    Both captured prefills agree: 16384 tokens launch the indexer 9 times and
    2358 tokens launch it twice, which is ``ceil((tokens + 1) / 2048)``.
    """
    chunks = max(math.ceil((t + 1) / 2048), 1)
    for layer in range(model.num_hidden_layers):
        ratio = model.compress_ratios[layer]
        kv = _kv_span(invocation.past_len, invocation.seq_len, ratio)
        q = (t, model.num_attention_heads // model.tp, head)
        launches.extend(
            _prefill_attention(
                model, layer, t, h, q_lora, q_b_out, q, kv, head, wo_in, o_lora,
                local_groups, wo_b_k, index_q_out, chunks,
            )
        )
        launches.extend(
            _prefill_moe(model, layer, t, h, inter, experts, routed)
        )
    launches.append(_launch("aiter::hc_head", "hc_head", ((t, h),), None))
    return launches


def _prefill_attention(
    model, layer, t, h, q_lora, q_b_out, q, kv, head, wo_in, o_lora,
    local_groups, wo_b_k, index_q_out, chunks,
) -> list[Launch]:
    ratio = model.compress_ratios[layer]
    block: list[Launch] = [
        _launch("aiter::mhc_pre_gemm_sqrsum", "mhc_sqrsum", ((t, h),), layer)
    ]
    if layer == 0:
        block.append(_launch("aiter::mhc_pre_big_fuse", "mhc_pre", ((t, h),), layer))
        block.append(_launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, h),), layer))
    else:
        block.append(
            _launch("aiter::mhc_pre_big_fuse_rmsnorm", "mhc_pre_rms", ((t, h),), layer)
        )
        block.append(
            _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((t, h),), layer)
        )
    # Prefill runs wq_a as one CK tile GEMM, not the decode split-K pair.
    block.append(_launch("aten::mm", "ck_tile", ((t, h), (h, q_lora)), layer))
    block.append(_launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, q_lora),), layer))
    block.append(_launch("aten::mm", "ck_gemm", ((t, q_lora), (q_lora, q_b_out)), layer))
    block.append(_launch("aiter::fuse_qk_norm_rope", "qk_rope", (q,), layer))
    if ratio == 4:
        block.extend(
            _prefill_c4(model, layer, t, h, q_lora, index_q_out, kv, chunks)
        )
    else:
        block.append(_launch("aten::mm", "tensile", ((t, h), (h, head)), layer))
        block.append(_launch("sglang::flash_c128", "c128", (q, (kv, 1, head)), layer))
        block.append(_launch("sglang::write_c128", "w128", ((kv, head),), layer))
    block.extend(
        [
            _launch("sglang::fused_norm_rope_flashmla", "flashmla", (q, (kv, 1, head)), layer),
            _launch("sglang::prefill_lengths", "pf_len", ((t,),), layer),
            _launch("sglang::prefill_indices", "pf_idx", ((t,),), layer),
            _launch("aiter::mla_decode_v4_asm", "mla", (q, (kv, 1, head)), layer),
            _launch("sglang::swa_scatter", "swa", (q,), layer),
            _launch("sglang::swa_scatter", "swa", (q,), layer),
            _launch("sglang::apply_rotary_emb", "rope", (q,), layer),
            _launch("aiter::dynamic_per_group_scaled_quant", "wo_quant", (q,), layer),
            _launch(
                "aiter::_opus_bmm_a8w8_mxscale_raw",
                "gemm_sf",
                ((local_groups, t, wo_in), (local_groups, wo_in, o_lora)),
                layer,
            ),
            _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((t, wo_b_k),), layer),
            _launch("aten::mm", "ck_gemm", ((t, wo_b_k), (wo_b_k, h)), layer),
            _launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer),
            _launch("aiter::mhc_post", "mhc_post", ((t, h),), layer),
        ]
    )
    return block


def _prefill_c4(model, layer, t, h, q_lora, index_q_out, kv, chunks) -> list[Launch]:
    index_dim = model.index_head_dim
    block = [
        _launch("aiter::gemm_a16w16", "hgemm", ((t, q_lora), (q_lora, index_q_out)), layer),
        _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((t, h),), layer),
        _launch("aten::mm", "ck_gemm", ((t, h), (h, model.index_n_heads)), layer),
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
            _launch("sglang::flash_c4", "c4", ((t, index_dim), (kv, 1, index_dim)), layer),
            _launch("sglang::write_c4", "w4", ((kv, index_dim),), layer),
        ]
    )
    return block


def _prefill_moe(model, layer, t, h, inter, experts, routed) -> list[Launch]:
    block = [
        _launch("aiter::mhc_pre_gemm_sqrsum", "mhc_sqrsum", ((t, h),), layer),
        _launch("aiter::mhc_pre_big_fuse_rmsnorm", "mhc_pre_rms", ((t, h),), layer),
        _launch("aten::mm", "gate", ((t, h), (h, experts)), layer),
    ]
    if layer < model.num_hash_layers:
        block.append(_launch("sglang::hash_topk", "hash_topk", ((t, experts),), layer))
    else:
        block.append(_launch("aiter::topk_gating_fwd", "topk", ((t, experts),), layer))
        block.append(
            _launch(
                "sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts",
                "shared",
                ((t, h), (h, inter)),
                layer,
            )
        )
    for _phase in range(4):
        block.append(_launch("aiter::moe_sorting_opus_fwd", "opus_sort", ((t, experts),), layer))
    block.extend(
        [
            _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((routed, h),), layer),
            _launch("aiter::mxfp4_moe_sort", "mxfp4", ((routed, h),), layer),
            _launch(
                "pseudo_op::moe_flydsl_stage1",
                "moe1",
                ((routed, h), (h, 2 * inter)),
                layer,
            ),
            _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((routed, inter),), layer),
            _launch("aiter::mxfp4_moe_sort", "mxfp4", ((routed, inter),), layer),
            _launch(
                "pseudo_op::moe_flydsl_stage2",
                "moe2",
                ((routed, inter), (inter, h)),
                layer,
            ),
            _launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer),
            _launch("aiter::mhc_post", "mhc_post", ((t, h),), layer),
        ]
    )
    return block


def _attention_block(
    model, layer, t, h, q_lora, q_b_out, q, kv, wo_in, o_lora, local_groups, wo_b_k, index_q_out
) -> list[Launch]:
    ratio = model.compress_ratios[layer]
    head = model.head_dim
    block: list[Launch] = []
    if layer == 0:
        block.append(_launch("aiter::mhc_pre_gemm_sqrsum", "mhc_sqrsum", ((t, h),), layer))
        block.append(_launch("aiter::mhc_pre_big_fuse", "mhc_pre", ((t, h),), layer))
        block.append(_launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, h),), layer))
    else:
        block.append(
            _launch("aiter::mhc_pre_big_fuse_rmsnorm", "mhc_pre_rms", ((t, h),), layer)
        )
        block.append(
            _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((t, h),), layer)
        )
    # wq_a is replicated. The split-K gemm launches the product and its reduce.
    wq_a = ((t, h), (h, q_lora))
    block.append(_launch("aiter::gemm_a8w8_blockscale_bpreshuffle", "gemm_a8w8", wq_a, layer))
    block.append(_launch("aiter::gemm_a8w8_blockscale_bpreshuffle", "gemm_red", wq_a, layer))
    block.append(_launch("aiter::fused_rms_fp8_quant", "rms_quant", ((t, q_lora),), layer))
    # wq_b is column-parallel on the attention group.
    wq_b = ((t, q_lora), (q_lora, q_b_out))
    block.append(_launch("aten::mm", "ck_gemm", wq_b, layer))
    block.append(_launch("aiter::fuse_qk_norm_rope", "qk_rope", (q,), layer))
    if ratio == 4:
        block.extend(_c4_indexer(model, layer, t, h, q_lora, index_q_out, kv))
    else:
        block.append(_launch("aten::mm", "tensile", ((t, h), (h, head)), layer))
        block.append(
            _launch("sglang::flash_c128", "c128", (q, (kv, 1, head)), layer)
        )
    block.append(
        _launch("sglang::fused_norm_rope_flashmla", "flashmla", (q, (kv, 1, head)), layer)
    )
    if ratio == 4:
        # The compressed tail is written after the flashmla norm, before MLA.
        block.append(
            _launch("sglang::fill_compress_tail", "fill_tail", ((kv, head),), layer)
        )
    block.append(_launch("aiter::mla_decode_v4_asm", "mla", (q, (kv, 1, head)), layer))
    block.append(_launch("aiter::mla_decode_stage2", "attn_asm", (q, (kv, 1, head)), layer))
    block.append(_launch("sglang::apply_rotary_emb", "rope", (q,), layer))
    block.append(
        _launch("aiter::dynamic_per_group_scaled_quant", "wo_quant", (q,), layer)
    )
    # wo_a is a grouped GEMM: one matrix per local attention group.
    wo_a = ((local_groups, t, wo_in), (local_groups, wo_in, o_lora))
    block.append(_launch("aiter::_opus_bmm_a8w8_mxscale_raw", "gemm_mx", wo_a, layer))
    block.append(
        _launch("aiter::dynamic_per_group_scaled_quant", "quant", ((t, wo_b_k),), layer)
    )
    # wo_b is row-parallel: this GPU multiplies its K slice into the full hidden.
    wo_b = ((t, wo_b_k), (wo_b_k, h))
    block.append(_launch("aten::mm", "ck_gemm", wo_b, layer))
    block.append(_launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer))
    block.append(
        _launch(
            "aiter::mhc_fused_post_pre_gemm_sqrsum",
            "mhc_fused",
            ((t, h),),
            layer,
        )
    )
    return block


def _c4_indexer(model, layer, t, h, q_lora, index_q_out, kv) -> list[Launch]:
    index_dim = model.index_head_dim
    return [
        _launch(
            "aiter::gemm_a16w16",
            "hgemm",
            ((t, q_lora), (q_lora, index_q_out)),
            layer,
        ),
        _launch(
            "aiter::dynamic_per_group_scaled_quant",
            "quant",
            ((t, h),),
            layer,
        ),
        _launch(
            "aten::mm",
            "ck_gemm",
            ((t, h), (h, model.index_n_heads)),
            layer,
        ),
        _launch(
            "aiter::rope_hadamard_rotate_activation_fp4quant",
            "rope_had",
            ((t, model.index_n_heads, index_dim),),
            layer,
        ),
        _launch("aten::mm", "tensile", ((t, h), (h, index_dim)), layer),
        _launch("sglang::flash_c4", "c4", ((t, index_dim), (kv, 1, index_dim)), layer),
        _launch(
            "aiter::norm_rope_hadamard_rotate_activation_fp4quant",
            "norm_rope_had",
            ((t, model.index_n_heads, index_dim),),
            layer,
        ),
        _launch(
            "sglang::pa_mqa_logits",
            "indexer",
            ((t, model.index_n_heads, index_dim), (kv, 1, index_dim)),
            layer,
        ),
        _launch(
            "sglang::topk",
            "topk_main",
            ((t, kv),),
            layer,
        ),
        _launch("aten::mm", "tensile", ((t, h), (h, index_dim)), layer),
        _launch("sglang::flash_c4", "c4", ((t, index_dim), (kv, 1, index_dim)), layer),
    ]


def _moe_block(model, layer, t, h, inter, experts, routed, last: bool) -> list[Launch]:
    block = [
        _launch("aiter::mhc_pre_big_fuse_rmsnorm", "mhc_pre_rms", ((t, h),), layer),
        _launch("aiter::gemm_a16w16", "hgemm", ((t, h), (h, experts)), layer),
    ]
    if layer < model.num_hash_layers:
        block.append(_launch("sglang::hash_topk", "hash_topk", ((t, experts),), layer))
    else:
        block.append(_launch("aiter::topk_gating_fwd", "topk", ((t, experts),), layer))
        block.append(
            _launch(
                "sglang_profiler::fused_moe_triton_kernels_fused_append_shared_experts",
                "shared",
                ((t, h), (h, inter)),
                layer,
            )
        )
    block.extend(
        [
            _launch("aiter::moe_sorting_fwd", "moe_sort", ((t, experts),), layer),
            _launch(
                "aiter::fused_dynamic_mx_quant_moe_sort_hip",
                "mx_sort",
                ((routed, h),),
                layer,
            ),
            _launch(
                "pseudo_op::moe_flydsl_stage1",
                "moe1",
                ((routed, h), (h, 2 * inter)),
                layer,
            ),
            _launch(
                "aiter::fused_dynamic_mx_quant_moe_sort_hip",
                "mx_sort",
                ((routed, inter),),
                layer,
            ),
            _launch(
                "pseudo_op::moe_flydsl_stage2",
                "moe2",
                ((routed, inter), (inter, h)),
                layer,
            ),
            _launch("aiter::cross_device_reduce", "allreduce", ((t, h),), layer),
        ]
    )
    if last:
        block.append(_launch("aiter::mhc_post", "mhc_post", ((t, h),), layer))
    else:
        block.append(
            _launch(
                "aiter::mhc_fused_post_pre_gemm_sqrsum",
                "mhc_fused",
                ((t, h),),
                layer,
            )
        )
    return block


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
