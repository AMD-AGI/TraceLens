###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Attention test harnesses for ``validate_perf_model``.

Includes ``make_varlen_seqlens`` / ``compute_varlen_annotation_stats``
which the parent ``validator.py`` also imports so the
perf-model runner sees the same ``c_sq`` / ``c_sqsq`` aggregates as the
test harness. These helpers are pure-Python and do NOT import torch at
module load time so the parent can use them on machines without torch
installed; ``torch`` is only imported inside each ``test_*`` function
body.

Every test function is parameterized over the dtype kwargs the kernel
actually consumes.
"""

import random

from TraceLens.TraceUtils.utils.annotation_utils import IterationAnnotation

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
        cuts = sorted(
            rng.sample(range(1, total_tokens), min(num_seqs - 1, total_tokens - 1))
        )
        boundaries = [0] + cuts + [total_tokens]
        seq_lengths = [
            boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)
        ]
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


def test__flash_attn_forward(
    seq_len,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    out_dtype="bf16",
    **_,
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
    print(
        f"test: flash_attn B={B} S={S} H_Q={H_Q} H_KV={H_KV} d={d} dtype={in_dtype}",
        flush=True,
    )
    q = torch.randn(B, S, H_Q, d, dtype=in_t, device=device)
    k = torch.randn(B, S, H_KV, d, dtype=in_t, device=device)
    v = torch.randn(B, S, H_KV, d, dtype=in_t, device=device)

    def call():
        return aiter.flash_attn_func(q, k, v, causal=True)

    return call


def test_wrapper_fmha_v3_fwd(
    seq_len,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    out_dtype="bf16",
    **_,
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
    softmax_scale = 1 / d**0.5
    device = "cuda"
    print(
        f"test: fmha_v3 B={B} S={S} H_Q={H_Q} H_KV={H_KV} d={d} dtype={in_dtype}",
        flush=True,
    )
    q = torch.randn(B, S, H_Q, d, device=device).to(in_t)
    k = torch.randn(B, S, H_KV, d, device=device).to(in_t)
    v = torch.randn(B, S, H_KV, d, device=device).to(in_t)

    def call():
        return fmha_v3_fwd(q, k, v, 0, softmax_scale, True, -1, -1, True, False, 1)

    return call


def test_mha_varlen_fwd(
    seq_len,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    out_dtype="bf16",
    varlen_seed=42,
    varlen_num_seqs=4,
    varlen_scenario="random",
    **_,
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
    softmax_scale = 1 / d**0.5
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

    def call():
        return mha_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            0,
            softmax_scale,
            0,
            False,
            True,
            -1,
            -1,
            0,
            True,
            False,
        )

    return call


def test_fmha_v3_varlen_fwd(
    seq_len,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    out_dtype="bf16",
    varlen_seed=42,
    varlen_num_seqs=4,
    varlen_scenario="random",
    **_,
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
    softmax_scale = 1 / d**0.5
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

    def call():
        return fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            0,
            softmax_scale,
            0,
            False,
            True,
            -1,
            -1,
            True,
            False,
            1,
        )

    return call


def test_vllm_unified_attention(
    seq_len=None,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    kv_dtype="fp8",
    out_dtype="bf16",
    annotation=None,
    num_decode_seqs=None,
    ctx_len=None,
    prefill_seq_len=0,
    **_,
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
    ann = IterationAnnotation(annotation) if annotation else None
    if ann is not None and ann.has_sqsk:
        # Requests within each phase are uniform: per-request lengths are the
        # phase sums divided by the request count.
        for n, sq, sk in (
            (ann.context_requests, ann.c_sq, ann.c_sk),
            (ann.generation_requests, ann.g_sq, ann.g_sk),
        ):
            seq_lens_q += [sq // n] * n if n else []
            seq_lens_k += [sk // n] * n if n else []
        prefill_seq_len = seq_lens_q[0] if ann.context_requests else 0
        num_decode_seqs = ann.generation_requests
        ctx_len = seq_lens_k[-1] if num_decode_seqs else 0
    else:
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

    softmax_scale = 1 / d**0.5
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

    block_table = torch.zeros(
        num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device
    )
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
            q=q,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    return _call


def test_unified_attention(
    seq_len,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    kv_dtype="fp8",
    out_dtype="bf16",
    annotation=None,
    **kwargs,
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
        seq_len=seq_len,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        in_dtype=in_dtype,
        kv_dtype=kv_dtype,
        out_dtype=out_dtype,
        annotation=annotation,
        num_decode_seqs=num_decode_seqs,
        ctx_len=ctx_len,
        prefill_seq_len=prefill_seq_len,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# DSV3 / DSV4 MLA attention harnesses
# ---------------------------------------------------------------------------


def test_dsv3_mla_decode_fwd(
    seq_len,
    E=64,
    num_heads_q=16,
    head_dim=576,
    page_size=1,
    kv_dtype="fp8",
    n_ctx=0,
    ctx_qlen=0,
    **_,
):
    """``aiter.mla.mla_decode_fwd`` (paged MLA attention through the MQA path).

    Reproduces ``pseudo_mla_decode_fwd``. The traces drive this entry point in
    two precisions and two batch shapes:

      * DSV3 decode          fp8 KV      ``mla_a8w8_qh16_qseqlen1_gqaratio16_ps``
      * GLM/Kimi decode      bf16 KV     ``mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps``
      * GLM chunked prefill  bf16 KV     same kernel, context + generation batch

    The scheduler expands a chunked prefill into one entry per query token, so a
    context request of ``ctx_qlen`` tokens contributes ``ctx_qlen`` entries whose
    kv lengths walk the causal ramp 1..ctx_qlen. That keeps ``max_seqlen_qo`` at
    1 for every batch shape, which is what the persistent-scheduler metadata
    (the ``_ps`` in the kernel name) is sized for.

    Parameters mapped from the trace:
      * ``--seq_len``     : per-sequence KV context length of a generation
                            request (g_sk / g_sq).
      * ``--E``           : number of generation requests (g_sq).
      * ``--n_ctx``       : number of context (prefill) requests, c_sq / ctx_qlen.
      * ``--ctx_qlen``    : query tokens per context request, c_sqsq / c_sq.
      * ``--kv_dtype``    : ``fp8`` or ``bf16``, taken from the traced KV cache
                            tensor dtype; it selects the a8w8 or a16w16 kernel.
      * ``--num_heads_q`` : number of Q heads (nhead_kv == 1 for MLA). Default 16.
      * ``--head_dim``    : qk_head_dim = kv_lora_rank + qk_rope_head_dim. Default 576.
    """
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1

    nhead = num_heads_q
    nhead_kv = 1
    qk_head_dim = head_dim  # 576 = 512 (kv_lora) + 64 (qk_rope)
    qk_rope_head_dim = 64
    kv_lora_rank = qk_head_dim - qk_rope_head_dim  # 512
    v_head_dim = kv_lora_rank  # absorbed decode: v_head_dim == kv_lora_rank
    device = "cuda"
    use_fp8 = str(kv_dtype).lower() in ("fp8", "fp8_e4m3", "float8_e4m3fn")
    causal = n_ctx > 0

    # One entry per query token: a causal ramp for each context request, then
    # the generation requests at their cached history length. Entries belonging
    # to the same request share one KV region, so the ramp re-reads the same
    # pages rather than touching fresh memory -- getting that wrong inflates the
    # measured byte count by the length of the ramp.
    kv_lens, kv_starts, region_base = [], [], 0
    for _ in range(int(n_ctx)):
        for length in range(1, int(ctx_qlen) + 1):
            kv_lens.append(length)
            kv_starts.append(region_base)
        region_base += int(ctx_qlen)
    for _ in range(int(E)):
        kv_lens.append(seq_len)
        kv_starts.append(region_base)
        region_base += seq_len
    batch_size = len(kv_lens)
    if batch_size == 0:
        raise ValueError("mla_decode_fwd harness needs at least one request")

    print(
        f"test: dsv3_mla_decode_fwd gen={E}x{seq_len} ctx={n_ctx}x{ctx_qlen} "
        f"entries={batch_size} heads={nhead} qk_d={qk_head_dim} v_d={v_head_dim} "
        f"kv_dtype={'fp8' if use_fp8 else 'bf16'} causal={causal} page={page_size}",
        flush=True,
    )

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(kv_lens, dtype=torch.int, device=device), 0
    )

    num_page = region_base + 128  # page_size == 1
    kv_indices = torch.cat(
        [
            torch.arange(start, start + length, dtype=torch.int, device=device)
            for start, length in zip(kv_starts, kv_lens)
        ]
    )
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device=device)

    # Every entry carries exactly one query token.
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int, device=device)
    total_q = batch_size
    max_seqlen_qo = 1

    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16, device=device)
    kv_buffer = torch.randn(
        (num_page * page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    if use_fp8:
        q, kv_buffer = q.to(dtypes.fp8), kv_buffer.to(dtypes.fp8)
        q_scale = torch.ones([1], dtype=torch.float, device=device)
        kv_scale = torch.ones([1], dtype=torch.float, device=device)
    else:
        q_scale = kv_scale = None
    sm_scale = 1.0 / (qk_head_dim**0.5)

    out = torch.empty(
        (total_q, nhead, v_head_dim), dtype=torch.bfloat16, device=device
    ).fill_(-1)

    # Persistent-scheduler work plan. get_mla_metadata_info_v1 returns the buffer
    # sizes as (ptrs, work_indptr, work_info_set, ...) while get_mla_metadata_v1
    # takes them as (ptrs, work_info_set, work_indptr, ...), so the two middle
    # buffers swap between allocation and the call, as vLLM does.
    specs = get_mla_metadata_info_v1(
        batch_size=batch_size,
        max_seqlen_qo=max_seqlen_qo,
        num_head_qo=nhead,
        q_dtype=q.dtype,
        kv_dtype=kv_buffer.dtype,
        is_sparse=False,
        fast_mode=True,
    )
    ptrs, work_indptr, work_info_set, red_indptr, red_final, red_partial = [
        torch.zeros(shape, dtype=dt, device=device) for shape, dt in specs
    ]
    get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        causal,
        ptrs,
        work_info_set,
        work_indptr,
        red_indptr,
        red_final,
        red_partial,
        page_size=page_size,
        kv_granularity=16,
        max_seqlen_qo=max_seqlen_qo,
        uni_seqlen_qo=max_seqlen_qo,
        fast_mode=True,
    )

    def _call():
        aiter.mla.mla_decode_fwd(
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
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
            work_meta_data=ptrs,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=red_indptr,
            reduce_final_map=red_final,
            reduce_partial_map=red_partial,
        )

    return _call
