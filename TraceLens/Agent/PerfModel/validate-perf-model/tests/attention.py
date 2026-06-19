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
}
