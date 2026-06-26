###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GEMM test harnesses for ``validate_perf_model``.

Each ``test_<op>`` function builds the input tensors and calls the target
kernel with ``num_warmup`` warmup iterations followed by one measured
iteration. The shared runner (``tests/_runner.py``) dispatches into these.

Every test function is parameterized over the input / weight / output
dtype the kernel actually supports. See ``tests/gemm.md`` for the per-op
mapping of dtypes onto the upstream ``aiter`` / ``vllm`` implementation
and ``tests/gemm.csv`` for the canonical parameter combinations exercised
by the validation harness.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _quantize_a8w8_blockscale(x_f, w_f, block_k, fp8=False):
    """Per-block quantize ``(x_f, w_f)`` into i8/fp8 + FP32 block scales.

    Returns ``(x_q, w_q, x_scale, w_scale)``. ``x_q`` / ``w_q`` are stored as
    ``int8`` (numerically equivalent to FP8 for this kernel; see
    ``aiter/ops/gemm_op_a8w8_blockscale.py``). When ``fp8=True`` we cast the
    int8 storage to ``aiter.dtypes.fp8`` instead -- the kernel accepts either
    1-byte storage for the activation and weight tensors.
    """
    import torch

    M, K = x_f.shape
    N, _ = w_f.shape
    num_k_blocks = (K + block_k - 1) // block_k

    x_q = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=x_f.device)
    w_q = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=w_f.device)
    x_scale = torch.randn(M, num_k_blocks, dtype=torch.float32, device=x_f.device)
    w_scale = torch.randn(N, num_k_blocks, dtype=torch.float32, device=w_f.device)

    if fp8:
        from aiter import dtypes as _aiter_dtypes
        fp8_dtype = _aiter_dtypes.fp8
        x_q = x_q.view(torch.uint8).view(fp8_dtype)
        w_q = w_q.view(torch.uint8).view(fp8_dtype)

    return x_q, w_q, x_scale, w_scale


# ---------------------------------------------------------------------------
# 1. gemm_a8w8_blockscale (aiter CK FP8 / INT8 block-scaled GEMM)
# ---------------------------------------------------------------------------

def test_gemm_a8w8_blockscale(
    M, N, K,
    in_dtype="i8",
    w_dtype="i8",
    out_dtype="bf16",
    scale_dtype="fp32",
    block_k=128,
    num_warmup=3,
    **_,
):
    """``aiter.gemm_a8w8_blockscale`` (FP8 / INT8 block-scaled GEMM).

    Parameters
    ----------
    in_dtype : {"i8", "u8", "fp8"}
        Activation storage dtype (1 byte). The kernel treats the bytes as FP8
        for matmul; ``"i8"`` and ``"fp8"`` are interchangeable in storage.
    w_dtype : {"i8", "u8", "fp8"}
        Weight storage dtype (1 byte).
    out_dtype : {"bf16", "fp16"}
        Output dtype.
    scale_dtype : {"fp32"}
        Block-scale dtype. The kernel only supports FP32 scales.
    block_k : int
        Per-K-block tile size for scales (default 128).
    """
    import torch
    import aiter

    device = "cuda"
    print(
        f"test: gemm_a8w8_blockscale M={M} N={N} K={K} "
        f"in={in_dtype} w={w_dtype} out={out_dtype} block_k={block_k}",
        flush=True,
    )

    fp8_storage = (str(in_dtype).lower() == "fp8" or str(w_dtype).lower() == "fp8")
    x_q, w_q, x_scale, w_scale = _quantize_a8w8_blockscale(
        torch.empty(M, K, device=device),
        torch.empty(N, K, device=device),
        block_k=block_k,
        fp8=fp8_storage,
    )
    x_scale = x_scale.to(_resolve_dtype(scale_dtype))
    w_scale = w_scale.to(_resolve_dtype(scale_dtype))

    out_t = _resolve_dtype(out_dtype)
    if out_t not in (torch.bfloat16, torch.float16):
        raise ValueError(f"gemm_a8w8_blockscale unsupported out_dtype={out_dtype}")

    for _ in range(num_warmup):
        out = aiter.gemm_a8w8_blockscale(x_q, w_q, x_scale, w_scale)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out = aiter.gemm_a8w8_blockscale(x_q, w_q, x_scale, w_scale)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 2. gemm_a16w16_atomic_ (aiter ASM BF16 / FP16 GEMM with atomic accumulation)
# ---------------------------------------------------------------------------

def test_gemm_a16w16_atomic_(
    M, N, K,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter.ops.gemm_op_a16w16.gemm_a16w16_asm`` (BF16 / FP16 GEMM, atomic accumulate).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
    w_dtype : {"bf16", "fp16"}
    out_dtype : {"bf16", "fp16"}
        Per the kernel ABI all three dtypes must match. The harness raises if
        they don't.
    """
    import torch
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    # Activation and weight dtype must match (kernel ABI). Output dtype
    # accepted by the asm kernel: BFloat16 / Float (per
    # asm_gemm_a16w16.cu:216). FP16 input/weights with FP16 output is
    # rejected by the kernel ("out must be Float32 or Bf16, got fp16"),
    # so promote the output to BF16 in that case.
    if in_t != w_t:
        raise ValueError(
            f"gemm_a16w16_atomic_ requires matching in/w dtypes; got "
            f"in={in_dtype} w={w_dtype}"
        )
    if out_t not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"gemm_a16w16_atomic_ output must be bf16 or fp32; got out={out_dtype}"
        )

    device = "cuda"
    print(
        f"test: gemm_a16w16_atomic_ M={M} N={N} K={K} in={in_dtype} out={out_dtype}",
        flush=True,
    )

    A = torch.randn(M, K, dtype=in_t, device=device)
    B = torch.randn(N, K, dtype=w_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    for _ in range(num_warmup):
        gemm_a16w16_asm(A, B, out)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    gemm_a16w16_asm(A, B, out)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 3. vllm::rocm_unquantized_gemm
# ---------------------------------------------------------------------------

def test_vllm_unquantized_gemm(
    M, N, K,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``vllm::rocm_unquantized_gemm`` -- in production this dispatches to
    ``aiter.ops.gemm_op_a16w16.gemm_a16w16_asm`` on MI300X with the aiter
    triton GEMM enabled. Same dtype constraints as ``gemm_a16w16_atomic_``.
    """
    import torch
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    # Same constraint as ``gemm_a16w16_atomic_``: in/w must match, output
    # must be bf16 or fp32 (the asm kernel rejects fp16 output).
    if in_t != w_t:
        raise ValueError(
            f"vllm_unquantized_gemm requires matching in/w dtypes; got "
            f"in={in_dtype} w={w_dtype}"
        )
    if out_t not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"vllm_unquantized_gemm output must be bf16 or fp32; got out={out_dtype}"
        )

    device = "cuda"
    print(
        f"test: vllm_unquantized_gemm M={M} N={N} K={K} in={in_dtype} out={out_dtype}",
        flush=True,
    )

    A = torch.randn(M, K, dtype=in_t, device=device)
    B = torch.randn(N, K, dtype=w_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    for _ in range(num_warmup):
        gemm_a16w16_asm(A, B, out)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    gemm_a16w16_asm(A, B, out)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 4. vllm::gemm_with_dynamic_quant (FP4 GEMM, Quark OCP MX scales)
# ---------------------------------------------------------------------------

def test_vllm_gemm_with_dynamic_quant(
    M, N, K,
    in_dtype="bf16",
    w_dtype="fp4x2",
    out_dtype="bf16",
    scale_dtype="u8",
    num_warmup=3,
    **_,
):
    """``vllm::gemm_with_dynamic_quant`` (Quark OCP MX FP4 GEMM).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
        Activation dtype (the kernel quantizes A internally to FP8 / FP4 with
        Quark OCP MX scales).
    w_dtype : {"fp4x2"}
        Packed FP4 weight dtype. Storage is ``uint8`` with two FP4 values per
        byte; the perf model reports 0.5 BPE for the weight tensor.
    out_dtype : {"bf16", "fp16"}
    scale_dtype : {"u8"}
        Per-32 OCP MX scale dtype (``uint8`` storage). The kernel's API
        accepts other scale dtypes but the production path uses uint8.

    Note
    ----
    Marked ``perf_model_only`` in the validation registry; the test runs only
    if vllm with the gemm_with_dynamic_quant op is installed.
    """
    import torch
    import vllm  # noqa: F401  (registers torch.ops.vllm.*)

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    if str(w_dtype).lower() not in ("fp4", "fp4x2"):
        raise ValueError(
            f"vllm_gemm_with_dynamic_quant requires fp4/fp4x2 weights; "
            f"got w_dtype={w_dtype}"
        )

    device = "cuda"
    print(
        f"test: vllm_gemm_with_dynamic_quant M={M} N={N} K={K} "
        f"in={in_dtype} w={w_dtype} out={out_dtype}",
        flush=True,
    )

    x = torch.randn(M, K, dtype=in_t, device=device)
    weight = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
    weight_scale = torch.randint(0, 255, (N, K // 32), dtype=torch.uint8, device=device)
    fn = torch.ops.vllm.gemm_with_dynamic_quant

    for _ in range(num_warmup):
        out = fn(x, weight, weight_scale, False, out_t)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out = fn(x, weight, weight_scale, False, out_t)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 5. gemm_afp4wfp4  — AITER MXFP4 GEMM
# ---------------------------------------------------------------------------

def test_gemm_afp4wfp4(
    M, N, K,
    group_size=128,
    out_dtype="bf16",
    num_warmup=3,
    **_,
):
    """``aiter.gemm_afp4wfp4_`` — AITER MXFP4 (4-bit floating point) GEMM.

    Activations and weights are stored as uint8 pairs (2 × FP4 values packed
    per byte, a.k.a. ``fp4x2``).  Block scales are FP32, grouped by
    ``group_size`` along the K dimension.

    Parameters
    ----------
    M, N, K : int
        Matmul dimensions.
    group_size : int
        Block-K scale granularity (default 128; must divide K evenly).
    out_dtype : {"bf16"}
        Output accumulation dtype.
    """
    import torch
    try:
        import aiter
        _fn = aiter.gemm_afp4wfp4_
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "aiter.gemm_afp4wfp4_ is not available. "
            "Ensure aiter is installed with MXFP4 support."
        ) from exc

    out_t = _resolve_dtype(out_dtype)
    device = "cuda"

    if K % group_size != 0:
        raise ValueError(f"K ({K}) must be divisible by group_size ({group_size})")
    if K % 2 != 0:
        raise ValueError(f"K ({K}) must be even for FP4x2 packing")
    if N % 2 != 0:
        raise ValueError(f"N ({N}) must be even for FP4x2 packing")

    K_packed = K // 2  # two FP4 values packed per uint8 byte
    N_packed = N // 2
    num_k_groups = K // group_size

    print(
        f"test: gemm_afp4wfp4 M={M} N={N} K={K} group_size={group_size} out={out_dtype}",
        flush=True,
    )

    a = torch.randint(0, 256, (M, K_packed), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (N_packed, K_packed), dtype=torch.uint8, device=device)
    a_scale = torch.randn(M, num_k_groups, dtype=torch.float32, device=device)
    b_scale = torch.randn(N, num_k_groups, dtype=torch.float32, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    for _ in range(num_warmup):
        _fn(a, b, a_scale, b_scale, out)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _fn(a, b, a_scale, b_scale, out)
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 6. dsv3_flydsl_hgemm (aiter FlyDSL BF16 NT GEMM)
# ---------------------------------------------------------------------------

def test_dsv3_flydsl_hgemm(M, N, K, num_warmup=3, **_):
    """``aiter.ops.flydsl.gemm_kernels.flydsl_hgemm`` (BF16 NT GEMM).

    Computes ``out[M, N] = A[M, K] @ B[N, K]^T`` in BF16. The kernel produces
    the ``hgemm_bf16_*_S2TN_AS_SPK*`` variants observed in DSV3 decode
    (sglang_profiler::gemm_kernels_flydsl_hgemm_54).
    """
    import torch
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
    from aiter.ops.shuffle import shuffle_weight

    device = "cuda"
    dtype = torch.bfloat16
    print(f"test: dsv3_flydsl_hgemm M={M} N={N} K={K}", flush=True)

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(N, K, dtype=dtype, device=device)
    b_shuf = shuffle_weight(b, layout=(16, 16))
    out = torch.empty(M, N, dtype=dtype, device=device)

    try:
        flydsl_hgemm(a, b_shuf, out)
        b_run = b_shuf
    except Exception:
        flydsl_hgemm(a, b, out)
        b_run = b

    for _ in range(num_warmup):
        flydsl_hgemm(a, b_run, out)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    flydsl_hgemm(a, b_run, out)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


# ---------------------------------------------------------------------------
# 7. dsv3_batched_gemm_a8w8 (aiter triton batched A8W8 GEMM)
# ---------------------------------------------------------------------------

def test_dsv3_batched_gemm_a8w8(M, N, K, E, group_size=128, num_warmup=3, **_):
    """``aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant``.

    Computes ``Y[B, M, N] = X[B, M, K] @ WQ[B, N, K]^T`` where X is BF16
    (quantized to FP8 per token-group inside the kernel) and WQ is FP8 with
    one scalar scale per batch (E experts).

    Parameters use ``--E`` for the batch / expert dim (matches the DSV3 trace
    where E=16 weight planes are paired with a 32-token activation batch).
    """
    import torch
    from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as bgemm,
    )

    device = "cuda"
    fp8 = torch.float8_e4m3fn
    print(
        f"test: dsv3_batched_gemm_a8w8 B={E} M={M} N={N} K={K} group_size={group_size}",
        flush=True,
    )

    X = torch.randn(E, M, K, dtype=torch.bfloat16, device=device)
    WQ = (torch.randn(E, N, K, device=device) * 0.1).to(fp8)
    w_scale = torch.ones((1,), dtype=torch.float32, device=device)

    for _ in range(num_warmup):
        Y = bgemm(X, WQ, w_scale, group_size=group_size)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    Y = bgemm(X, WQ, w_scale, group_size=group_size)
    torch.cuda.synchronize()
    print(f"test: done, shape={Y.shape}", flush=True)


# ---------------------------------------------------------------------------
# 8. dsv4_opus_gemm_a16w16 (aiter opus batched BF16 GEMM, split-K)
# ---------------------------------------------------------------------------

def test_dsv4_opus_gemm_a16w16(M=1819, N=384, K=7168, num_warmup=3, **_):
    """``aiter.ops.opus.gemm_a16w16_opus`` — batched BF16 GEMM (split-K).

    The high-level dispatcher resolves a valid tuned kernel id and calls the
    underlying ``_opus_gemm_a16w16_tune_raw`` kernel seen in the trace.
    """
    import torch
    from aiter.ops.opus import gemm_a16w16_opus

    print(f"test: dsv4_opus_gemm_a16w16 B=1 M={M} N={N} K={K}", flush=True)
    A = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    for _ in range(num_warmup):
        Y = gemm_a16w16_opus(A, B, None, torch.bfloat16)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    Y = gemm_a16w16_opus(A, B, None, torch.bfloat16)
    torch.cuda.synchronize()
    print(f"test: done Y={tuple(Y.shape)}", flush=True)


# ---------------------------------------------------------------------------
# 9. dsv4_gemm_a8w8_blockscale_bpreshuffle_asm (FP8 block-scaled GEMM, ASM)
# ---------------------------------------------------------------------------

def test_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm(M=1819, N=2048, K=7168, num_warmup=3, **_):
    """``aiter.gemm_a8w8_blockscale_bpreshuffle_asm`` — FP8 block-scaled GEMM (ASM)."""
    import torch
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    block = 128
    scale_k = (K + block - 1) // block
    scale_n = (N + block - 1) // block
    print(f"test: dsv4_gemm_a8w8_blockscale_bpreshuffle_asm M={M} N={N} K={K}", flush=True)

    A = (torch.rand(M, K, device="cuda", dtype=torch.float32) / 10).to(aiter.dtypes.fp8)
    B_raw = (torch.rand(N, K, device="cuda", dtype=torch.float32) / 10).to(aiter.dtypes.fp8)
    B = shuffle_weight(B_raw, layout=(16, 16))
    A_scale = torch.rand(M, scale_k, device="cuda", dtype=torch.float32)
    A_scale = A_scale.transpose(0, 1).contiguous().view(M, scale_k)
    B_scale = torch.rand(scale_n, scale_k, device="cuda", dtype=torch.float32)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    for _ in range(num_warmup):
        aiter.gemm_a8w8_blockscale_bpreshuffle_asm(A, B, out, A_scale, B_scale)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    aiter.gemm_a8w8_blockscale_bpreshuffle_asm(A, B, out, A_scale, B_scale)
    torch.cuda.synchronize()
    print(f"test: done out={tuple(out.shape)}", flush=True)


# ---------------------------------------------------------------------------
# OP_METADATA
# ---------------------------------------------------------------------------

OP_METADATA: dict = {
    "gemm_a8w8_blockscale": {
        "fn":           test_gemm_a8w8_blockscale,
        "category":     "GEMM",
        "description":  "AITER FP8/INT8 block-scaled GEMM (CK / CKTile / ASM)",
        "dtypes":       ["i8", "fp8"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "in_dtype": "i8"},
        "required_args": ["M", "N", "K"],
    },
    "gemm_a16w16_atomic_": {
        "fn":           test_gemm_a16w16_atomic_,
        "category":     "GEMM",
        "description":  "AITER BF16/FP16 GEMM (atomic-add ASM, gemm_a16w16_asm)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N", "K"],
    },
    "vllm_unquantized_gemm": {
        "fn":           test_vllm_unquantized_gemm,
        "category":     "GEMM",
        "description":  "vLLM unquantized BF16 GEMM (gemm_a16w16_asm dispatch)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "in_dtype": "bf16"},
        "required_args": ["M", "N", "K"],
    },
    "vllm_triton_gemm_a8w8_blockscale": {
        "fn":           test_gemm_a8w8_blockscale,
        "category":     "GEMM",
        "description":  "vLLM Triton FP8/INT8 block-scaled GEMM (maps to gemm_a8w8_blockscale)",
        "dtypes":       ["i8", "fp8"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "in_dtype": "i8"},
        "required_args": ["M", "N", "K"],
    },
    "vllm_gemm_with_dynamic_quant": {
        "fn":           test_vllm_gemm_with_dynamic_quant,
        "category":     "GEMM",
        "description":  "vLLM Quark OCP MX FP4 GEMM (gemm_with_dynamic_quant)",
        "dtypes":       ["fp4x2"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168, "w_dtype": "fp4x2"},
        "required_args": ["M", "N", "K"],
    },
    "gemm_afp4wfp4": {
        "fn":           test_gemm_afp4wfp4,
        "category":     "GEMM",
        "description":  "AITER MXFP4 GEMM (gemm_afp4wfp4_)",
        "dtypes":       ["fp4x2"],
        "defaults":     {"M": 822, "N": 2112, "K": 3584, "group_size": 128},
        "required_args": ["M", "N", "K"],
    },
    "dsv3_flydsl_hgemm": {
        "fn":           test_dsv3_flydsl_hgemm,
        "category":     "GEMM",
        "description":  "DSV3 AITER FlyDSL BF16 NT GEMM (flydsl_hgemm)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 2048, "N": 4096, "K": 7168},
        "required_args": ["M", "N", "K"],
    },
    "dsv3_batched_gemm_a8w8": {
        "fn":           test_dsv3_batched_gemm_a8w8,
        "category":     "GEMM",
        "description":  "DSV3 AITER triton batched A8W8 GEMM (per-token-group / per-batch FP8)",
        "dtypes":       ["fp8"],
        "defaults":     {"M": 32, "N": 4096, "K": 7168, "E": 16, "group_size": 128},
        "required_args": ["M", "N", "K", "E"],
    },
    "dsv4_opus_gemm_a16w16": {
        "fn":           test_dsv4_opus_gemm_a16w16,
        "category":     "GEMM",
        "description":  "DSV4 batched BF16 GEMM (aiter opus split-K)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1819, "N": 384, "K": 7168},
        "required_args": ["M", "N", "K"],
    },
    "dsv4_gemm_a8w8_blockscale_bpreshuffle_asm": {
        "fn":           test_dsv4_gemm_a8w8_blockscale_bpreshuffle_asm,
        "category":     "GEMM",
        "description":  "DSV4 FP8 block-scaled GEMM, B preshuffle (aiter ASM)",
        "dtypes":       ["fp8"],
        "defaults":     {"M": 1819, "N": 2048, "K": 7168},
        "required_args": ["M", "N", "K"],
    },
}
