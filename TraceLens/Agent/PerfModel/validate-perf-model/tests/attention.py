###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attention test harnesses for ``validate_perf_model``.

Includes ``make_varlen_seqlens`` / ``compute_varlen_annotation_stats``
which the parent ``validate_perf_model.py`` also imports so the
perf-model runner sees the same ``c_sq`` / ``c_sqsq`` aggregates as the
test harness. These helpers are pure-Python and do NOT import torch at
module load time so the parent can use them on machines without torch
installed; ``torch`` is only imported inside each ``test_*`` function
body.

Every test function is parameterized over the dtype kwargs the kernel
actually consumes. See ``tests/attention.md`` for per-op coverage and
``tests/attention.csv`` for the canonical parameter combinations.
"""

import random

from ._dtypes import resolve_dtype as _resolve_dtype

VARLEN_SCENARIOS = ("random", "mixed_prefill_decode")


def make_varlen_seqlens(total_tokens, num_seqs=4, seed=42, scenario="random"):
    """Generate variable-length seq partitioning for a varlen attention call.

    Returns ``(seq_lengths_q, cu_q, seq_lengths_k, cu_k)``. For ``scenario``:

    * ``"random"`` -- self-attention varlen: Q == K and a random partition
      of ``total_tokens`` across ``num_seqs`` chunks (deterministic for a
      given seed). For backwards compat, ``cu_q[-1] == cu_k[-1] ==
      total_tokens`` and ``seq_lengths_q == seq_lengths_k``.
    * ``"mixed_prefill_decode"`` -- one prefill sequence with
      ``s_q == s_k == total_tokens`` and ``num_seqs - 1`` decode sequences
      with ``s_q == 1`` and ``s_k == total_tokens`` (i.e. all sequences
      share the same KV-cache length while only the prefill contributes
      a multi-token query). Total Q tokens =
      ``total_tokens + num_seqs - 1``; total K/V tokens =
      ``total_tokens * num_seqs``.

    Deterministic for a given ``seed`` (only the random scenario uses it)
    so the parent validator and the rocprof'd child see the same layout.
    """
    if scenario not in VARLEN_SCENARIOS:
        raise ValueError(
            f"unknown varlen scenario {scenario!r}; expected one of {VARLEN_SCENARIOS}"
        )
    if num_seqs < 1:
        raise ValueError(f"num_seqs must be >= 1, got {num_seqs}")

    if scenario == "random":
        rng = random.Random(seed)
        cuts = sorted(rng.sample(range(1, total_tokens), min(num_seqs - 1, total_tokens - 1)))
        boundaries = [0] + cuts + [total_tokens]
        seq_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        cu_seqlens = [0]
        for sl in seq_lengths:
            cu_seqlens.append(cu_seqlens[-1] + sl)
        return (seq_lengths, cu_seqlens, list(seq_lengths), list(cu_seqlens))

    # mixed_prefill_decode
    seq_lengths_q = [total_tokens] + [1] * (num_seqs - 1)
    seq_lengths_k = [total_tokens] * num_seqs
    cu_q = [0]
    for s in seq_lengths_q:
        cu_q.append(cu_q[-1] + s)
    cu_k = [0]
    for s in seq_lengths_k:
        cu_k.append(cu_k[-1] + s)
    return (seq_lengths_q, cu_q, seq_lengths_k, cu_k)


def compute_varlen_annotation_stats(seq_lengths_q, seq_lengths_k=None):
    """Return ``(c_sq, c_sqsk)`` aggregates for a varlen attention call.

    * ``c_sq``  = ``sum(seq_lengths_q)`` -- drives Q + output HBM traffic.
    * ``c_sqsk`` = ``sum(sq * sk)`` -- drives FLOPs.

    Backwards compat: if ``seq_lengths_k`` is omitted the result matches the
    historical ``(c_sq=Sigma s, c_sqsq=Sigma s^2)`` self-attention aggregates.
    """
    if seq_lengths_k is None:
        seq_lengths_k = seq_lengths_q
    if len(seq_lengths_q) != len(seq_lengths_k):
        raise ValueError(
            f"seq_lengths_q and seq_lengths_k must have equal length, "
            f"got {len(seq_lengths_q)} vs {len(seq_lengths_k)}"
        )
    c_sq = sum(seq_lengths_q)
    c_sqsk = sum(sq * sk for sq, sk in zip(seq_lengths_q, seq_lengths_k))
    return (c_sq, c_sqsk)


def _parse_unified_attention_annotation(annotation):
    """Parse an ``execute_..._context_..._generation_(...)`` iter marker.

    Returns a dict with ctx_/gen_ ``req``, ``sq``, ``sk``, ``sqsq``, ``sqsk``
    fields, or ``None`` if the annotation does not match the vLLM convention.
    """
    if not annotation:
        return None
    import re

    pat = re.compile(
        r"execute_(?P<iter>\d+)_context_(?P<ctx_req>\d+)\(sq(?P<ctx_sq>\d+)sk(?P<ctx_sk>\d+)"
        r"sqsq(?P<ctx_sqsq>\d+)sqsk(?P<ctx_sqsk>\d+)\)_generation_(?P<gen_req>\d+)\("
        r"sq(?P<gen_sq>\d+)sk(?P<gen_sk>\d+)sqsq(?P<gen_sqsq>\d+)sqsk(?P<gen_sqsk>\d+)\)"
    )
    m = pat.search(str(annotation))
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def test__flash_attn_forward(
    seq_len, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", out_dtype="bf16", num_warmup=3, **_,
):
    """``aiter._flash_attn_forward`` via ``aiter.flash_attn_func`` (CK).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
        Dtype of Q, K, V (kernel requires all three to match).
    out_dtype : {"bf16", "fp16"}
        Output dtype (typically matches input).

    Note
    ----
    The upstream perf-model class (``aiter__flash_attn_forward``) hardcodes
    a 2-byte-per-element assumption in its ``bytes()`` method, so changing
    these dtypes only affects the test runner side. The kwargs are
    accepted for parity with other harnesses.
    """
    import aiter
    import torch

    in_t = _resolve_dtype(in_dtype)
    _resolve_dtype(out_dtype)
    B = 1
    S, H_Q, H_KV, d = seq_len, num_heads_q, num_heads_kv, head_dim
    device = "cuda"
    print(f"test: flash_attn B={B} S={S} H_Q={H_Q} H_KV={H_KV} d={d} dtype={in_dtype}", flush=True)
    q = torch.randn(B, S, H_Q, d, dtype=in_t, device=device)
    k = torch.randn(B, S, H_KV, d, dtype=in_t, device=device)
    v = torch.randn(B, S, H_KV, d, dtype=in_t, device=device)
    for _ in range(num_warmup):
        out = aiter.flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out = aiter.flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_wrapper_fmha_v3_fwd(
    seq_len, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", out_dtype="bf16", num_warmup=3, **_,
):
    """``aiter::wrapper_fmha_v3_fwd`` via ``aiter.ops.mha.fmha_v3_fwd``.

    Same dtype constraints as :func:`test__flash_attn_forward`.
    """
    import torch
    from aiter.ops.mha import fmha_v3_fwd

    in_t = _resolve_dtype(in_dtype)
    _resolve_dtype(out_dtype)
    B = 1
    S, H_Q, H_KV, d = seq_len, num_heads_q, num_heads_kv, head_dim
    softmax_scale = 1 / d ** 0.5
    device = "cuda"
    print(f"test: fmha_v3 B={B} S={S} H_Q={H_Q} H_KV={H_KV} d={d} dtype={in_dtype}", flush=True)
    q = torch.randn(B, S, H_Q, d, device=device).to(in_t)
    k = torch.randn(B, S, H_KV, d, device=device).to(in_t)
    v = torch.randn(B, S, H_KV, d, device=device).to(in_t)
    for _ in range(num_warmup):
        out, lse, S_dmask, _ = fmha_v3_fwd(q, k, v, 0, softmax_scale, True, -1, -1, True, False, 1)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out, lse, S_dmask, _ = fmha_v3_fwd(q, k, v, 0, softmax_scale, True, -1, -1, True, False, 1)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_mha_varlen_fwd(
    seq_len, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", out_dtype="bf16", num_warmup=3,
    varlen_seed=42, varlen_num_seqs=4, varlen_scenario="random", **_,
):
    """``aiter::mha_varlen_fwd`` (variable-length flash attention forward, CK).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
        Q/K/V dtype.
    out_dtype : {"bf16", "fp16"}
        Output dtype (typically matches input).
    varlen_num_seqs : int, default 4
        Number of sequences packed into the varlen call.
    varlen_scenario : {"random", "mixed_prefill_decode"}, default "random"
        ``random`` partitions ``seq_len`` tokens across ``varlen_num_seqs``
        random self-attention chunks. ``mixed_prefill_decode`` builds one
        prefill sequence (Q==K==``seq_len``) plus ``varlen_num_seqs - 1``
        decode sequences (Q=1, K=``seq_len`` each) -- a chunked-prefill
        batch.
    """
    import torch
    from aiter.ops.mha import mha_varlen_fwd

    in_t = _resolve_dtype(in_dtype)
    _resolve_dtype(out_dtype)
    S, H_Q, H_KV, d = seq_len, num_heads_q, num_heads_kv, head_dim
    sq, cu_q_list, sk, cu_k_list = make_varlen_seqlens(
        S, num_seqs=varlen_num_seqs, seed=varlen_seed, scenario=varlen_scenario
    )
    total_q = cu_q_list[-1]
    total_kv = cu_k_list[-1]
    max_seqlen_q = max(sq)
    max_seqlen_k = max(sk)
    min_seqlen_q = min(sq)
    softmax_scale = 1 / d ** 0.5
    device = "cuda"
    print(
        f"test: mha_varlen scenario={varlen_scenario} num_seqs={len(sq)} "
        f"total_q={total_q} total_kv={total_kv} H_Q={H_Q} H_KV={H_KV} d={d} "
        f"max_q={max_seqlen_q} max_k={max_seqlen_k} dtype={in_dtype}",
        flush=True,
    )
    q = torch.randn(total_q, H_Q, d, device=device).to(in_t)
    k = torch.randn(total_kv, H_KV, d, device=device).to(in_t)
    v = torch.randn(total_kv, H_KV, d, device=device).to(in_t)
    cu_q = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    cu_k = torch.tensor(cu_k_list, dtype=torch.int32, device=device)
    for _ in range(num_warmup):
        out, lse, S_dmask, _ = mha_varlen_fwd(
            q, k, v, cu_q, cu_k, max_seqlen_q, max_seqlen_k, min_seqlen_q,
            0, softmax_scale, 0, False, True, -1, -1, 0, True, False,
        )
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out, lse, S_dmask, _ = mha_varlen_fwd(
        q, k, v, cu_q, cu_k, max_seqlen_q, max_seqlen_k, min_seqlen_q,
        0, softmax_scale, 0, False, True, -1, -1, 0, True, False,
    )
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_fmha_v3_varlen_fwd(
    seq_len, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", out_dtype="bf16", num_warmup=3,
    varlen_seed=42, varlen_num_seqs=4, varlen_scenario="random", **_,
):
    """``aiter::fmha_v3_varlen_fwd`` (variable-length FMHA v3 forward).

    Same dtype constraints and ``varlen_scenario`` semantics as
    :func:`test_mha_varlen_fwd`.
    """
    import torch
    from aiter.ops.mha import fmha_v3_varlen_fwd

    in_t = _resolve_dtype(in_dtype)
    _resolve_dtype(out_dtype)
    S, H_Q, H_KV, d = seq_len, num_heads_q, num_heads_kv, head_dim
    sq, cu_q_list, sk, cu_k_list = make_varlen_seqlens(
        S, num_seqs=varlen_num_seqs, seed=varlen_seed, scenario=varlen_scenario
    )
    total_q = cu_q_list[-1]
    total_kv = cu_k_list[-1]
    max_seqlen_q = max(sq)
    max_seqlen_k = max(sk)
    min_seqlen_q = min(sq)
    softmax_scale = 1 / d ** 0.5
    device = "cuda"
    print(
        f"test: fmha_v3_varlen scenario={varlen_scenario} num_seqs={len(sq)} "
        f"total_q={total_q} total_kv={total_kv} H_Q={H_Q} H_KV={H_KV} d={d} "
        f"max_q={max_seqlen_q} max_k={max_seqlen_k} dtype={in_dtype}",
        flush=True,
    )
    q = torch.randn(total_q, H_Q, d, device=device).to(in_t)
    k = torch.randn(total_kv, H_KV, d, device=device).to(in_t)
    v = torch.randn(total_kv, H_KV, d, device=device).to(in_t)
    cu_q = torch.tensor(cu_q_list, dtype=torch.int32, device=device)
    cu_k = torch.tensor(cu_k_list, dtype=torch.int32, device=device)
    for _ in range(num_warmup):
        out, lse, S_dmask, _ = fmha_v3_varlen_fwd(
            q, k, v, cu_q, cu_k, max_seqlen_q, max_seqlen_k, min_seqlen_q,
            0, softmax_scale, 0, False, True, -1, -1, True, False, 1,
        )
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out, lse, S_dmask, _ = fmha_v3_varlen_fwd(
        q, k, v, cu_q, cu_k, max_seqlen_q, max_seqlen_k, min_seqlen_q,
        0, softmax_scale, 0, False, True, -1, -1, True, False, 1,
    )
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_vllm_unified_attention(
    seq_len=None, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", kv_dtype="fp8", out_dtype="bf16", num_warmup=3,
    annotation=None, num_decode_seqs=None, ctx_len=None, prefill_seq_len=0, **_,
):
    """``vllm::unified_attention_with_output`` via the aiter Triton unified_attention kernel.

    Uses ``aiter.ops.triton.attention.unified_attention.unified_attention`` --
    the same varlen Triton kernel that vLLM ROCm calls for
    ``vllm::unified_attention_with_output``.  This kernel handles prefill and
    decode requests in a **single call** via a packed Q tensor and
    ``cu_seqlens_q``.  Three test scenarios are supported:

    1. **Single decode** (``num_decode_seqs=1``): one outstanding request,
       query length 1, attending to ``ctx_len`` KV tokens.
    2. **Batch decode** (``num_decode_seqs=N``): N decode requests, each with
       query length 1 and ``ctx_len`` KV tokens.
    3. **Mixed prefill+decode** (``prefill_seq_len>0``): one causal prefill
       request (query length = KV length = ``prefill_seq_len``) followed by
       ``num_decode_seqs`` decode requests (query length 1 each, KV length
       ``ctx_len`` each).  The packed Q tensor is
       ``[prefill_seq_len + num_decode_seqs, H_Q, d]``.

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
        Query / output dtype.
    kv_dtype : {"fp8", "bf16", "fp16"}
        Paged KV-cache dtype.
    out_dtype : {"bf16", "fp16"}
        Output dtype (typically matches ``in_dtype``).
    num_decode_seqs : int, optional
        Explicit decode request count (Q=1 each). Falls back to ``seq_len``.
    ctx_len : int, optional
        KV-cache tokens per decode request. Falls back to ``seq_len``.
    prefill_seq_len : int, optional
        Length of the prefill sequence included in the same kernel call.
        When 0 (default) the batch is decode-only.
    """
    import torch
    from aiter.ops.triton.attention.unified_attention import unified_attention

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    kv_t = _resolve_dtype(kv_dtype)

    H_Q, H_KV, d = num_heads_q, num_heads_kv, head_dim
    block_size = 16
    device = "cuda"

    if num_decode_seqs is None:
        num_decode_seqs = seq_len if seq_len is not None else 1
    if ctx_len is None:
        ctx_len = seq_len if seq_len is not None else 256
    prefill_seq_len = int(prefill_seq_len or 0)

    # Per-request query / kv lengths (prefill first, then decodes).
    seq_lens_q = []
    seq_lens_k = []
    if prefill_seq_len > 0:
        seq_lens_q.append(prefill_seq_len)
        seq_lens_k.append(prefill_seq_len)
    for _i in range(num_decode_seqs):
        seq_lens_q.append(1)
        seq_lens_k.append(ctx_len)

    num_seqs = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    max_seqlen_q = max(seq_lens_q)
    max_seqlen_k = max(seq_lens_k)

    cu_q_list = [0]
    for s in seq_lens_q:
        cu_q_list.append(cu_q_list[-1] + s)

    softmax_scale = 1 / d ** 0.5
    print(
        f"test: vllm_unified_attention num_seqs={num_seqs} total_q={total_q} "
        f"prefill={prefill_seq_len} decodes={num_decode_seqs} ctx_len={ctx_len} "
        f"H_Q={H_Q} H_KV={H_KV} d={d} in={in_dtype} kv={kv_dtype}",
        flush=True,
    )

    # Packed query tensor [total_q, H_Q, d].
    q = torch.randn(total_q, H_Q, d, device=device).to(in_t)
    out = torch.empty(total_q, H_Q, d, device=device, dtype=out_t)

    # Paged KV cache: enough blocks for the longest per-request kv length.
    max_blocks_per_seq = (max_seqlen_k + block_size - 1) // block_size
    total_blocks = max_blocks_per_seq * num_seqs + 1
    key_cache = torch.randn(total_blocks, block_size, H_KV, d, device=device).to(kv_t)
    value_cache = torch.randn(total_blocks, block_size, H_KV, d, device=device).to(kv_t)

    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)
    blk = 0
    for i in range(num_seqs):
        nblk = (seq_lens_k[i] + block_size - 1) // block_size
        for j in range(nblk):
            block_table[i, j] = blk
            blk = (blk + 1) % total_blocks
    seqused_k = torch.tensor(seq_lens_k, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(cu_q_list, dtype=torch.int32, device=device)

    k_descale = None
    v_descale = None
    if kv_dtype == "fp8":
        k_descale = torch.ones(num_seqs, H_KV, device=device, dtype=torch.float32)
        v_descale = torch.ones(num_seqs, H_KV, device=device, dtype=torch.float32)

    def _call():
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale, causal=True,
            window_size=(-1, -1), block_table=block_table,
            softcap=0.0, q_descale=None, k_descale=k_descale, v_descale=v_descale,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_unified_attention(
    seq_len, num_heads_q=32, num_heads_kv=8, head_dim=128,
    in_dtype="bf16", kv_dtype="fp8", out_dtype="bf16", num_warmup=3,
    annotation=None, **kwargs,
):
    """``aiter`` triton ``unified_attention`` forward (paged KV / varlen).

    ``aiter::unified_attention`` and ``vllm::unified_attention_with_output``
    both dispatch to the same aiter Triton kernel.  This harness delegates
    directly to :func:`test_vllm_unified_attention`, treating ``seq_len`` as
    ``num_decode_seqs`` (with ``ctx_len = seq_len``) when neither
    ``num_decode_seqs`` nor ``ctx_len`` is supplied by the caller.

    Dtype kwargs ``in_dtype`` / ``kv_dtype`` / ``out_dtype`` are forwarded
    verbatim.
    """
    num_decode_seqs = kwargs.pop("num_decode_seqs", None)
    ctx_len = kwargs.pop("ctx_len", None)
    prefill_seq_len = kwargs.pop("prefill_seq_len", 0)
    if num_decode_seqs is None:
        num_decode_seqs = seq_len
    if ctx_len is None:
        ctx_len = seq_len
    return test_vllm_unified_attention(
        seq_len=seq_len, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
        head_dim=head_dim, in_dtype=in_dtype, kv_dtype=kv_dtype, out_dtype=out_dtype,
        num_warmup=num_warmup, annotation=annotation,
        num_decode_seqs=num_decode_seqs, ctx_len=ctx_len, prefill_seq_len=prefill_seq_len,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# DSV3 / DSV4 MLA attention harnesses
# ---------------------------------------------------------------------------

def _build_mla_ps_metadata(*, seq_len, batch_size, num_heads, qk_head_dim,
                           v_head_dim, block_size, is_causal, dtype_q, dtype_kv,
                           device):
    """Build the planner outputs + Q/K/V tensors needed by mla_prefill_ps_asm_fwd / mla_reduce_v1.

    Mirrors the canonical setup in ``../aiter/op_tests/test_mla_prefill_ps.py``.
    Returns a dict with every tensor the two kernels need so the harnesses can
    pull from one place.
    """
    import torch
    import aiter
    from aiter import dtypes, per_tensor_quant

    assert num_heads >= 1
    gqa_ratio = num_heads // num_heads
    softmax_scale = 1 / (qk_head_dim ** 0.5)
    tile_q = 256
    tile_kv = 128
    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity
    kvlen_granularity = max(tile_kv, block_size)

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    seq_lens_kv = torch.full((batch_size,), seq_len, dtype=torch.int, device=device)
    seq_lens_qo = seq_lens_kv.clone()
    max_qlen = int(seq_lens_qo.max().item())
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    num_blocks = int(kv_indptr[-1].item())
    kv_indices = torch.randint(0, num_blocks, (num_blocks,), dtype=torch.int, device=device)
    num_tokens = int(qo_indptr[-1].item())

    Q_bf16 = torch.randn(num_tokens, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    K_bf16 = torch.randn(num_blocks, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    V_bf16 = K_bf16[:, :, :v_head_dim].contiguous()

    q_quant, q_scale = per_tensor_quant(Q_bf16, quant_dtype=dtype_q)
    k_quant, k_scale = per_tensor_quant(K_bf16, quant_dtype=dtype_kv)
    v_quant, v_scale = per_tensor_quant(V_bf16, quant_dtype=dtype_kv)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_heads,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
    )

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device)

    aiter.get_ps_metadata_v1(
        qo_indptr.cpu(), kv_indptr.cpu(), seq_lens_kv.cpu(),
        gqa_ratio, num_heads,
        work_metadata_ptrs, work_indptr, work_info,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=block_size,
        is_causal=is_causal,
    )
    torch.cuda.synchronize()

    output = torch.empty(num_tokens, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)
    logits = torch.empty(
        reduce_partial_map.size(0) * tile_q, num_heads, v_head_dim,
        dtype=dtypes.fp32, device=device,
    )
    attn_lse = torch.empty(
        reduce_partial_map.size(0) * tile_q, num_heads,
        dtype=dtypes.fp32, device=device,
    )
    final_lse = torch.empty(num_tokens, num_heads, dtype=dtypes.fp32, device=device)

    return dict(
        q_quant=q_quant,
        k_quant=k_quant,
        v_quant=v_quant,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        work_indptr=work_indptr,
        work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        output=output,
        logits=logits,
        attn_lse=attn_lse,
        final_lse=final_lse,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        max_qlen=max_qlen,
        softmax_scale=softmax_scale,
        tile_q=tile_q,
    )


def test_dsv3_mla_prefill_ps_asm_fwd(seq_len, E=16, num_heads_q=16, head_dim=192,
                                     block_size=1, num_warmup=3, **_):
    """``aiter.mla_prefill_ps_asm_fwd`` (FP8 MLA prefill, persistent scheduler ASM).

    Parameters mapped from the DSV3 trace:
      * ``--seq_len``  : per-batch context length (==Q_len for causal prefill).
      * ``--E``        : batch size (number of sequences). Defaults to 16.
      * ``--num_heads_q`` : number of Q heads (also num_heads_kv). Default 16.
      * ``--head_dim`` : qk_head_dim (D_lora + D_pe). Default 192. v_head_dim = head_dim - 64.

    Only runs on gfx950 (the only arch the ASM kernel ships for).
    """
    import torch
    import aiter
    from aiter import dtypes

    v_head_dim = head_dim - 64
    print(
        f"test: dsv3_mla_prefill_ps_asm_fwd ctx={seq_len} batch={E} "
        f"heads={num_heads_q} qk_d={head_dim} v_d={v_head_dim} block={block_size}",
        flush=True,
    )
    meta = _build_mla_ps_metadata(
        seq_len=seq_len,
        batch_size=E,
        num_heads=num_heads_q,
        qk_head_dim=head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
        is_causal=True,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
        device="cuda",
    )

    def _call():
        aiter.mla_prefill_ps_asm_fwd(
            meta["q_quant"], meta["k_quant"], meta["v_quant"],
            meta["qo_indptr"], meta["kv_indptr"], meta["kv_indices"],
            meta["work_indptr"], meta["work_info"],
            meta["max_qlen"], meta["softmax_scale"], True,
            meta["logits"], meta["attn_lse"], meta["output"],
            meta["q_scale"], meta["k_scale"], meta["v_scale"],
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done, output={meta['output'].shape}", flush=True)


def test_dsv3_mla_reduce_v1(seq_len, E=16, num_heads_q=16, head_dim=192,
                            block_size=1, num_warmup=3, **_):
    """``aiter.mla_reduce_v1`` (cross-split MLA reduce).

    Shares the persistent-scheduler planner with mla_prefill_ps_asm_fwd. We
    run prefill once to populate the partial logits / LSE, then time the
    reduce kernel.
    """
    import torch
    import aiter
    from aiter import dtypes

    v_head_dim = head_dim - 64
    print(
        f"test: dsv3_mla_reduce_v1 ctx={seq_len} batch={E} "
        f"heads={num_heads_q} qk_d={head_dim} v_d={v_head_dim}",
        flush=True,
    )
    meta = _build_mla_ps_metadata(
        seq_len=seq_len,
        batch_size=E,
        num_heads=num_heads_q,
        qk_head_dim=head_dim,
        v_head_dim=v_head_dim,
        block_size=block_size,
        is_causal=True,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
        device="cuda",
    )
    aiter.mla_prefill_ps_asm_fwd(
        meta["q_quant"], meta["k_quant"], meta["v_quant"],
        meta["qo_indptr"], meta["kv_indptr"], meta["kv_indices"],
        meta["work_indptr"], meta["work_info"],
        meta["max_qlen"], meta["softmax_scale"], True,
        meta["logits"], meta["attn_lse"], meta["output"],
        meta["q_scale"], meta["k_scale"], meta["v_scale"],
    )
    torch.cuda.synchronize()

    def _call():
        aiter.mla_reduce_v1(
            meta["logits"], meta["attn_lse"], meta["reduce_indptr"],
            meta["reduce_final_map"], meta["reduce_partial_map"],
            meta["tile_q"], meta["output"], meta["final_lse"],
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done, output={meta['output'].shape}", flush=True)


def test_dsv3_mla_decode_fwd(seq_len, E=64, num_heads_q=16, head_dim=576,
                             page_size=1, num_warmup=3, **_):
    """``aiter.mla.mla_decode_fwd`` (FP8 paged MLA decode, q_len==1).

    Reproduces ``pseudo_mla_decode_fwd`` from the DSV3 decode trace: a batch of
    decode steps (one query token each) attending over an FP8 paged KV cache.
    The pseudo op aggregates the ASM core kernel
    (``mla_a8w8_qh16_qseqlen1_gqaratio16_ps``) plus the cross-split reduce; the
    auto-discovered kernel filter measures the dominant (core) kernel.

    Parameters mapped from the trace:
      * ``--seq_len``     : per-sequence KV context length (g_sk / batch).
      * ``--E``           : batch size (number of decode sequences). Default 64.
      * ``--num_heads_q`` : number of Q heads (nhead_kv == 1 for MLA). Default 16.
      * ``--head_dim``    : qk_head_dim = kv_lora_rank + qk_rope_head_dim. Default 576.
    """
    import torch
    import aiter
    from aiter import dtypes

    batch_size = E
    nhead = num_heads_q
    nhead_kv = 1
    qk_head_dim = head_dim               # 576 = 512 (kv_lora) + 64 (qk_rope)
    qk_rope_head_dim = 64
    kv_lora_rank = qk_head_dim - qk_rope_head_dim   # 512
    v_head_dim = kv_lora_rank            # absorbed decode: v_head_dim == kv_lora_rank
    ctx_lens = seq_len
    device = "cuda"

    print(
        f"test: dsv3_mla_decode_fwd ctx={ctx_lens} batch={batch_size} "
        f"heads={nhead} qk_d={qk_head_dim} v_d={v_head_dim} page={page_size}",
        flush=True,
    )

    seq_lens_kv = torch.full((batch_size,), ctx_lens, dtype=torch.int, device=device)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens_kv, dim=0)
    total_kv = int(kv_indptr[-1].item())

    num_page = total_kv + 128            # page_size == 1
    kv_indices = torch.arange(total_kv, dtype=torch.int, device=device)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device=device)

    # Decode: exactly one query token per sequence.
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int, device=device)
    total_q = batch_size
    max_seqlen_qo = 1

    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16, device=device)
    kv_buffer = torch.randn(
        (num_page * page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16, device=device,
    )

    q_fp8 = q.to(dtypes.fp8)
    kv_fp8 = kv_buffer.to(dtypes.fp8)
    q_scale = torch.ones([1], dtype=torch.float, device=device)
    kv_scale = torch.ones([1], dtype=torch.float, device=device)
    sm_scale = 1.0 / (qk_head_dim ** 0.5)

    out = torch.empty((total_q, nhead, v_head_dim), dtype=torch.bfloat16, device=device).fill_(-1)

    def _call():
        aiter.mla.mla_decode_fwd(
            q_fp8,
            kv_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done, output={out.shape}", flush=True)


def test_dsv4_pa_sparse_prefill_opus(
    M=1819, num_heads_q=32, head_dim=512,
    total_pages=329728, total_tokens=1819,
    nnz_prefix=2095488, nnz_extend=232832,
    num_warmup=3, **_,
):
    """``aiter.pa_sparse_prefill_opus`` — two-region sparse paged prefill MLA."""
    import math
    import torch
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    N, H, D = M, num_heads_q, head_dim
    print(f"test: dsv4_pa_sparse_prefill_opus N={N} H={H} D={D} "
          f"nnz_prefix={nnz_prefix} nnz_extend={nnz_extend}", flush=True)

    def _csr(num_rows, pool_rows, target_nnz, seed):
        # Even split per query row; sample indices with replacement on GPU
        # (duplicates are fine for HW-counter / latency measurement).
        base = target_nnz // num_rows
        rem = target_nnz % num_rows
        lens = torch.full((num_rows,), base, dtype=torch.int32, device="cuda")
        if rem:
            lens[:rem] += 1
        indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device="cuda")
        indptr[1:] = torch.cumsum(lens, dim=0)
        nnz = int(indptr[-1].item())
        g = torch.Generator(device="cuda"); g.manual_seed(seed)
        indices = torch.randint(0, pool_rows, (nnz,), dtype=torch.int32,
                                device="cuda", generator=g)
        return indptr, indices

    q = (torch.randn(N, H, D, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)
    unified_kv = (torch.randn(total_pages, D, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)
    kv = (torch.randn(total_tokens, D, device="cuda", dtype=torch.float32) * 0.5).to(torch.bfloat16)
    attn_sink = torch.randn(H, device="cuda", dtype=torch.float32) * 0.25

    kv_indptr_prefix, kv_indices_prefix = _csr(N, total_pages, nnz_prefix, 1)
    kv_indptr_extend, kv_indices_extend = _csr(N, total_tokens, nnz_extend, 2)
    softmax_scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)

    def _call():
        pa_sparse_prefill_opus(
            q, unified_kv,
            kv_indices_prefix, kv_indptr_prefix,
            kv, kv_indices_extend, kv_indptr_extend,
            attn_sink, softmax_scale, out,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done out={tuple(out.shape)}", flush=True)


OP_METADATA: dict = {
    "_flash_attn_forward": {
        "fn": test__flash_attn_forward,
        "category": "InferenceAttention",
        "description": "AITER FlashAttention-2 forward (aiter.flash_attn_func)",
        "dtypes": ["bf16", "fp16"],
        "defaults": {"seq_len": 1024, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": ["seq_len"],
    },
    "wrapper_fmha_v3_fwd": {
        "fn": test_wrapper_fmha_v3_fwd,
        "category": "InferenceAttention",
        "description": "AITER FMHA v3 fixed-length forward (causal decode)",
        "dtypes": ["bf16", "fp8"],
        "defaults": {"seq_len": 1024, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": ["seq_len"],
    },
    "mha_varlen_fwd": {
        "fn": test_mha_varlen_fwd,
        "category": "InferenceAttention",
        "description": "AITER variable-length MHA forward (packed Q/K/V)",
        "dtypes": ["bf16", "fp16"],
        "defaults": {"seq_len": 1024, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": ["seq_len"],
    },
    "fmha_v3_varlen_fwd": {
        "fn": test_fmha_v3_varlen_fwd,
        "category": "InferenceAttention",
        "description": "AITER FMHA v3 variable-length forward",
        "dtypes": ["bf16", "fp8"],
        "defaults": {"seq_len": 1024, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": ["seq_len"],
    },
    "unified_attention": {
        "fn": test_unified_attention,
        "category": "InferenceAttention",
        "description": "AITER unified paged-decode attention (aiter.unified_attention)",
        "dtypes": ["bf16", "fp8"],
        "defaults": {"seq_len": 256, "num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": ["seq_len"],
    },
    "vllm_unified_attention": {
        "fn": test_vllm_unified_attention,
        "category": "InferenceAttention",
        "description": "vLLM unified paged-decode attention (torch.ops.vllm.unified_attention)",
        "defaults": {"num_heads_q": 32, "num_heads_kv": 8, "head_dim": 128, "in_dtype": "bf16"},
        "required_args": [],
        "test_cases": [
            {"num_decode_seqs": 1, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 1, "ctx_len": 512, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1, "ctx_len": 512, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 1, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 128, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 128, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 128, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 128, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 512, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 512, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 512, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 512, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 1024, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1024, "ctx_len": 256, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 1024, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1024, "ctx_len": 1024, "prefill_seq_len": 0, "kv_dtype": "bf16"},
            {"num_decode_seqs": 128, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "fp8"},
            {"num_decode_seqs": 128, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "bf16"},
            {"num_decode_seqs": 512, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "fp8"},
            {"num_decode_seqs": 512, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "bf16"},
            {"num_decode_seqs": 1024, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "fp8"},
            {"num_decode_seqs": 1024, "ctx_len": 512, "prefill_seq_len": 1024, "kv_dtype": "bf16"},
        ],
    },
    "dsv3_mla_prefill_ps_asm_fwd": {
        "fn": test_dsv3_mla_prefill_ps_asm_fwd,
        "category": "InferenceAttention",
        "description": "DSV3 AITER ASM MLA prefill (persistent scheduler, FP8)",
        "dtypes": ["fp8"],
        "defaults": {"seq_len": 2048, "E": 16, "num_heads_q": 16, "head_dim": 192, "block_size": 1},
        "required_args": ["seq_len"],
    },
    "dsv3_mla_reduce_v1": {
        "fn": test_dsv3_mla_reduce_v1,
        "category": "InferenceAttention",
        "description": "DSV3 AITER MLA cross-split reduce (paired with prefill)",
        "dtypes": ["fp8"],
        "defaults": {"seq_len": 2048, "E": 16, "num_heads_q": 16, "head_dim": 192, "block_size": 1},
        "required_args": ["seq_len"],
    },
    "dsv4_pa_sparse_prefill_opus": {
        "fn": test_dsv4_pa_sparse_prefill_opus,
        "category": "InferenceAttention",
        "description": "DSV4 sparse paged prefill MLA attention (aiter)",
        "dtypes": ["bf16"],
        "defaults": {"M": 1819, "num_heads_q": 32, "head_dim": 512},
        "required_args": ["M", "num_heads_q", "head_dim"],
    },
}
