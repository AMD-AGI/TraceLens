###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU harnesses for Kimi-K3 ops authored in kimi_k3_perf_model_extension.py."""

from ._dtypes import resolve_dtype as _resolve_dtype


def test_kimi_situ_and_mul(M=7, N=768, num_warmup=3, **_):
    """``torch.ops._C.situ_and_mul`` (Kimi SituGLU)."""
    import torch
    from vllm import _custom_ops as ops  # noqa: F401  loads _C

    device = "cuda"
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    inp = torch.randn(M, 2 * N, dtype=torch.bfloat16, device=device)
    beta, linear_beta = 4.0, 25.0
    print(f"test: _C.situ_and_mul M={M} N={N}", flush=True)
    for _ in range(num_warmup):
        torch.ops._C.situ_and_mul(out, inp, beta, linear_beta)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    torch.ops._C.situ_and_mul(out, inp, beta, linear_beta)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_kimi_static_per_tensor_quant(M=7, N=6912, num_warmup=3, **_):
    """``aiter.static_per_tensor_quant``."""
    import torch
    import aiter

    device = "cuda"
    inp = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    out = torch.empty(M, N, dtype=_resolve_dtype("fp8"), device=device)
    scale = torch.tensor([0.1], dtype=torch.float32, device=device)
    print(f"test: aiter.static_per_tensor_quant M={M} N={N}", flush=True)
    for _ in range(num_warmup):
        aiter.static_per_tensor_quant(out, inp, scale)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    aiter.static_per_tensor_quant(out, inp, scale)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_kimi_aten_addmm_(M=7, N=896, K=3584, num_warmup=3, **_):
    """In-place ``aten::addmm_``."""
    import torch

    device = "cuda"
    C = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    print(f"test: aten.addmm_ M={M} N={N} K={K}", flush=True)
    for _ in range(num_warmup):
        C.addmm_(A, B)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    C.addmm_(A, B)
    torch.cuda.synchronize()
    print(f"test: done, shape={C.shape}", flush=True)


def test_kimi_moe_sorting_opus_fwd(M=7, topk=16, E=896, block_m=32, num_warmup=3, **_):
    """``aiter.moe_sorting_opus_fwd`` at the Kimi-K3 decode shape."""
    import torch
    import aiter

    device = "cuda"
    num_experts = int(E)
    unit_size = int(block_m)
    topk_ids = torch.randint(
        0, num_experts, (M, topk), dtype=torch.int32, device=device
    )
    topk_weights = torch.rand(M, topk, dtype=torch.float32, device=device)
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * unit_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + unit_size - 1) // unit_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((0, 0), dtype=torch.bfloat16, device=device)
    ws_size = aiter.moe_sorting_opus_get_workspace_size(M, num_experts, topk, 0)
    workspace = (
        torch.empty(ws_size, dtype=torch.uint8, device=device) if ws_size > 0 else None
    )
    print(
        f"test: moe_sorting_opus_fwd M={M} topk={topk} E={num_experts} block={unit_size}",
        flush=True,
    )

    def _call():
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            unit_size,
            None,
            None,
            workspace,
            0,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print("test: done", flush=True)


def test_kimi_fused_kda_decode(
    seq_len=7, num_heads_q=12, head_dim=128, E=2732, num_warmup=3, **_
):
    """``ops.fused_kda_decode`` — conv + KDA recurrence + gated RMSNorm.

    ``E`` is the KV-page-pool depth (default 2732, the traced Kimi-K3 TP=8
    value). The pool must stay production-sized and each iteration must read a
    disjoint slot set: 94% of the roofline bytes are fp32 recurrent-state
    read+write, and a small pool with fixed indices keeps that state resident
    in L2 across iterations, under-counting HBM traffic by ~15%.
    """
    import torch
    from vllm import _custom_ops as ops

    T = int(seq_len)
    H = int(num_heads_q)
    D = int(head_dim)
    W = 4
    device = "cuda"
    dim = H * D
    slots = max(int(E), T)
    packed_x = torch.randn(T, 3 * dim, dtype=torch.bfloat16, device=device)
    weight = 0.1 * torch.randn(3, W, dim, dtype=torch.float32, device=device)
    # SD cache layout: [slots, W-1, 3*dim] then transpose to [slots, 3*dim, W-1]
    conv_state = 0.1 * torch.randn(
        slots, W - 1, 3 * dim, dtype=torch.bfloat16, device=device
    ).transpose(1, 2)
    raw_g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device)
    raw_beta = torch.randn(1, T, H, dtype=torch.bfloat16, device=device)
    output_gate = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(D, dtype=torch.float32, device=device)
    A_log = 0.5 * torch.randn(H, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(dim, dtype=torch.float32, device=device)
    state = 0.01 * torch.randn(slots, H, D, D, dtype=torch.float32, device=device)
    state_mb = state.numel() * state.element_size() / 2**20
    print(
        f"test: fused_kda_decode T={T} H={H} D={D} pages={slots} "
        f"state={state_mb:.0f} MiB",
        flush=True,
    )

    # Walk a fresh window of the pool per call so the measured iteration sees
    # cold state pages, as it does in production where ~60 other layers run
    # between two decode steps of the same layer.
    windows = max(1, slots // T)
    index_sets = [
        torch.arange(i * T, i * T + T, dtype=torch.int32, device=device)
        for i in range(min(windows, num_warmup + 1))
    ]

    def _call(state_indices):
        return ops.fused_kda_decode(
            x=packed_x,
            weight=weight,
            bias=None,
            conv_state=conv_state,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            state=state,
            lower_bound=-5.0,
            output_gate=output_gate,
            norm_weight=norm_weight,
            norm_eps=1e-5,
        )

    for i in range(num_warmup):
        _call(index_sets[i % len(index_sets)])
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out = _call(index_sets[num_warmup % len(index_sets)])
    torch.cuda.synchronize()
    print(f"test: done, out={tuple(out.shape)}", flush=True)


def test_kimi_gather_and_maybe_dequant_cache(
    seq_len=1024, head_dim=576, num_warmup=3, **_
):
    """``_C_cache_ops.gather_and_maybe_dequant_cache`` FP8 MLA gather."""
    import torch
    from vllm import _custom_ops as ops

    device = "cuda"
    entry_size = int(head_dim)
    total_tokens = int(seq_len)
    block_size = 64
    batch_size = 1
    num_blocks = max(16, (total_tokens + block_size - 1) // block_size + 2)
    kv_cache_dtype = "fp8"
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    src_cache = torch.randint(
        0, 256, (num_blocks, block_size, entry_size), device=device, dtype=torch.uint8
    ).view(torch.float8_e4m3fn)
    seq_len_tensor = torch.tensor([total_tokens], dtype=torch.int32, device=device)
    cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seq_lens[1:] = seq_len_tensor
    token_to_seq = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device), seq_len_tensor
    )
    block_table = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=device
    )
    block_table[0] = torch.arange(num_blocks, dtype=torch.int32, device=device)
    dst = torch.zeros((total_tokens, entry_size), dtype=torch.bfloat16, device=device)
    print(
        f"test: gather_and_maybe_dequant_cache tokens={total_tokens} d={entry_size}",
        flush=True,
    )

    def _call():
        ops.gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            total_tokens,
            kv_cache_dtype,
            scale,
            None,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print(f"test: done, dst={tuple(dst.shape)}", flush=True)
