###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FusedRoPE-family test harnesses for ``validate_perf_model``.

Covers two RoPE implementations:

  * ``aiter::rope_cached_positions_2c_fwd_impl`` — two-channel (Q + K) cached
    positions forward RoPE from the aiter package. Base class:
    ``aiter_rope_cached_positions_2c_fwd_impl(FusedRoPE)``.

  * ``sgl_kernel::rotary_embedding`` — SGLang rotate-half in-place RoPE on Q
    and K tensors. Base class: ``sgl_kernel_rotary_embedding(FusedRoPE)``.

Both ops are memory-bandwidth-bound elementwise transforms, so the dominant
roofline metric is HBM read+write of the Q/K tensors.

See ``tests/rope.md`` (to be generated) for the per-op dtype/shape matrix and
``test_cases/rope.csv`` for the canonical parameter combinations.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


# ---------------------------------------------------------------------------
# 1. aiter rope_cached_positions_2c_fwd_impl  (aiter pattern)
# ---------------------------------------------------------------------------

def test_rope_cached_positions_2c_fwd_impl(
    M,
    num_heads_q=16,
    num_heads_kv=1,
    head_dim=64,
    in_dtype="bf16",
    rotate_style=0,
    num_warmup=3,
    **_,
):
    """``aiter.ops.rope.rope_cached_positions_2c_fwd_impl`` — 2-channel RoPE.

    Allocates Q and K input/output buffers plus cos/sin tables and a position
    index tensor, then runs the aiter NEOX-style cached-positions RoPE kernel.

    Parameters
    ----------
    M : int
        Number of tokens (sequence length * batch, flattened).
    num_heads_q : int
        Query head count per token (default 16).
    num_heads_kv : int
        Key/value head count per token (default 1 for MLA/GQA).
    head_dim : int
        Head dimension in elements (default 64).
    in_dtype : {"bf16", "fp16"}
        Input / output storage dtype; cossin cache uses same dtype.
    rotate_style : int
        Rotation style: 0 = NEOX, 1 = GPT-J (default 0).
    """
    import torch
    try:
        from aiter.ops.rope import rope_cached_positions_2c_fwd_impl as _fn
    except ImportError:
        from aiter import rope_cached_positions_2c_fwd_impl as _fn

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    d = head_dim
    d_cs = d // 2  # reuse_freqs_front_part=True; rotated half of each head
    max_pos = max(M * 8, 4096)
    B, S = 1, M

    print(
        f"test: rope_cached_positions_2c_fwd_impl "
        f"M={M} H_q={num_heads_q} H_kv={num_heads_kv} d={d} dtype={in_dtype}",
        flush=True,
    )

    # input / output tensors
    input_x = torch.randn(B, S, num_heads_q, d, dtype=in_t, device=device)
    input_y = torch.randn(B, S, num_heads_kv, d, dtype=in_t, device=device)
    output_x = torch.empty_like(input_x)
    output_y = torch.empty_like(input_y)

    # cos/sin caches  (max_pos, 1, 1, d_cs)
    cos_cache = torch.randn(max_pos, 1, 1, d_cs, dtype=in_t, device=device)
    sin_cache = torch.randn(max_pos, 1, 1, d_cs, dtype=in_t, device=device)

    # per-token position indices
    positions = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)

    reuse_freqs_front_part = True
    nope_first = False

    for _ in range(num_warmup):
        _fn(
            output_x, output_y,
            input_x, input_y,
            cos_cache, sin_cache,
            positions,
            rotate_style, reuse_freqs_front_part, nope_first,
        )
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _fn(
        output_x, output_y,
        input_x, input_y,
        cos_cache, sin_cache,
        positions,
        rotate_style, reuse_freqs_front_part, nope_first,
    )
    torch.cuda.synchronize()
    print(
        f"test: done, output_x={output_x.shape} output_y={output_y.shape}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 2. sgl_kernel rotary_embedding  (SGLang pattern)
# ---------------------------------------------------------------------------

def test_sgl_kernel_rotary_embedding(
    M,
    num_heads_q=32,
    num_heads_kv=8,
    head_dim=128,
    in_dtype="bf16",
    is_neox=True,
    num_warmup=3,
    **_,
):
    """``sgl_kernel.rotary_embedding`` — in-place rotate-half RoPE on Q and K.

    The SGLang RoPE kernel operates in-place on flattened Q and K buffers of
    shape ``[n_tokens, n_heads * head_dim]``, using a combined cos/sin cache
    of shape ``[max_pos, rot_dim]`` where ``rot_dim = head_dim``.

    Parameters
    ----------
    M : int
        Number of tokens.
    num_heads_q : int
        Query head count (default 32).
    num_heads_kv : int
        Key/value head count (default 8).
    head_dim : int
        Head dimension in elements (default 128).
    in_dtype : {"bf16", "fp16"}
        Q/K storage dtype.
    is_neox : bool
        NEOX-style rotation if True (default), GPT-J style if False.
    """
    import torch
    try:
        import sgl_kernel
        _fn = sgl_kernel.rotary_embedding
    except ImportError as exc:
        raise ImportError(
            "sgl_kernel is not installed. Install it with: pip install sgl-kernel"
        ) from exc

    in_t = _resolve_dtype(in_dtype)
    device = "cuda"
    rot_dim = head_dim  # full head rotation
    max_pos = max(M * 8, 4096)

    print(
        f"test: sgl_kernel_rotary_embedding "
        f"M={M} H_q={num_heads_q} H_kv={num_heads_kv} d={head_dim} "
        f"dtype={in_dtype} neox={is_neox}",
        flush=True,
    )

    positions = torch.arange(M, dtype=torch.int64, device=device)
    query = torch.randn(M, num_heads_q * head_dim, dtype=in_t, device=device)
    key = torch.randn(M, num_heads_kv * head_dim, dtype=in_t, device=device)
    # cos/sin packed: shape [max_pos, rot_dim]
    cos_sin_cache = torch.randn(max_pos, rot_dim, dtype=in_t, device=device)

    for _ in range(num_warmup):
        _fn(positions, query, key, head_dim, cos_sin_cache, is_neox)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _fn(positions, query, key, head_dim, cos_sin_cache, is_neox)
    torch.cuda.synchronize()
    print(f"test: done, query={query.shape} key={key.shape}", flush=True)


# ---------------------------------------------------------------------------
# 3. dsv3_fused_qk_rope_cat_and_cache_mla  (aiter triton MLA RoPE + KV cache)
# ---------------------------------------------------------------------------

def test_dsv3_fused_qk_rope_cat_and_cache_mla(M, num_heads_q=16, head_dim=576,
                                              group_size=64, num_warmup=3, **_):
    """``aiter.ops.triton.fusions.fused_kv_cache.fused_qk_rope_cat_and_cache_mla``.

    Constructs the smallest realistic MLA prefill/decode call:
      * ``--M``         : number of tokens being processed
      * ``--num_heads_q``: query heads (H_q)
      * ``--head_dim``  : total head dim D = D_lora + D_pe; defaults to 576 (DSV3)
      * ``--group_size``: D_pe (rotary dim); defaults to 64 (DSV3)

    The kv_cache buffer is sized to ``T + 4096`` so the random slot_mapping
    has room. Uses BF16 throughout (matches the DSV3 trace).
    """
    import torch
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla

    device = "cuda"
    dtype = torch.bfloat16
    T = M
    H_q = num_heads_q
    D_pe = group_size
    D_lora = head_dim - D_pe
    if D_lora <= 0:
        raise ValueError(f"head_dim ({head_dim}) must exceed group_size/D_pe ({D_pe})")
    KH = 1
    num_kv_cache_tokens = max(T + 4096, 8192)
    max_pos = max(T * 16, 4096)
    print(
        f"test: dsv3_fused_qk_rope_cat_and_cache_mla T={T} H_q={H_q} "
        f"D_lora={D_lora} D_pe={D_pe} num_kv_cache_tokens={num_kv_cache_tokens}",
        flush=True,
    )

    q_nope = torch.randn(T, H_q, D_lora, dtype=dtype, device=device)
    q_pe = torch.randn(T, H_q, D_pe, dtype=dtype, device=device)
    k_nope = torch.randn(T, KH, D_lora, dtype=dtype, device=device)
    k_pe = torch.randn(T, KH, D_pe, dtype=dtype, device=device)
    kv_cache = torch.zeros(num_kv_cache_tokens, KH, D_lora + D_pe, dtype=dtype, device=device)
    pos = torch.randint(0, max_pos, (T,), dtype=torch.int64, device=device)
    slot_mapping = torch.randperm(num_kv_cache_tokens, device=device, dtype=torch.int64)[:T]
    cos = torch.randn(max_pos, 1, 1, D_pe // 2, dtype=dtype, device=device)
    sin = torch.randn(max_pos, 1, 1, D_pe // 2, dtype=dtype, device=device)
    k_scale = torch.ones((1,), dtype=torch.float32, device=device)[0]

    def _call():
        return fused_qk_rope_cat_and_cache_mla(
            q_nope, q_pe, k_nope, k_pe, kv_cache, slot_mapping, pos, cos, sin, k_scale,
            is_neox=True, num_decode_toks_for_zeros=0, apply_scale=False,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print("test: done", flush=True)


# ---------------------------------------------------------------------------
# OP_METADATA
# ---------------------------------------------------------------------------

#: Metadata for all FusedRoPE harnesses in this module.
OP_METADATA: dict = {
    "rope_cached_positions_2c_fwd_impl": {
        "fn":           test_rope_cached_positions_2c_fwd_impl,
        "category":     "FusedRoPE",
        "description":  "AITER 2-channel cached-positions forward RoPE",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {
            "M": 822,
            "num_heads_q": 16,
            "num_heads_kv": 1,
            "head_dim": 64,
            "in_dtype": "bf16",
        },
        "required_args": ["M"],
    },
    "sgl_kernel_rotary_embedding": {
        "fn":           test_sgl_kernel_rotary_embedding,
        "category":     "FusedRoPE",
        "description":  "SGLang in-place rotate-half RoPE (Q and K)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {
            "M": 1024,
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "head_dim": 128,
            "in_dtype": "bf16",
        },
        "required_args": ["M"],
    },
    "dsv3_fused_qk_rope_cat_and_cache_mla": {
        "fn":           test_dsv3_fused_qk_rope_cat_and_cache_mla,
        "category":     "FusedRoPE",
        "description":  "DSV3 AITER triton MLA RoPE + KV-cache fusion",
        "dtypes":       ["bf16"],
        "defaults":     {
            "M": 32,
            "num_heads_q": 16,
            "head_dim": 576,
            "group_size": 64,
        },
        "required_args": ["M"],
    },
}
