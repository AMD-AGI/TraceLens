###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Activation / quantization test harnesses for ``validate_perf_model``.

The "other" category bucket covers gated activations (silu/gelu *and_mul*),
the aiter dynamic per-token FP8 quant, and the vLLM Triton group-quant
variant that don't naturally fit gemm/moe/attention/rmsnorm.

See ``tests/other.md`` for the per-op mapping of dtypes onto the upstream
``aiter`` / ``vllm`` implementation and ``tests/other.csv`` for the
canonical parameter combinations.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


# ---------------------------------------------------------------------------
# Activation _and_mul kernels (SwiGLU-style: input [M, 2N] -> out [M, N])
# ---------------------------------------------------------------------------

def _activation_and_mul(
    fn_name, M, N,
    in_dtype="bf16",
    out_dtype="bf16",
    num_warmup=3,
):
    """Generic ``aiter.<fn_name>(out, inp)`` driver for SwiGLU-style activations.

    Allocates ``inp`` of shape ``[M, 2 * N]`` (gate || up) and ``out`` of
    shape ``[M, N]`` then invokes ``aiter.<fn_name>`` with the (out, inp)
    signature used by aiter's elementwise activations.
    """
    import torch
    import aiter

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)

    device = "cuda"
    print(
        f"test: {fn_name} M={M} N={N} in={in_dtype} out={out_dtype}",
        flush=True,
    )
    fn = getattr(aiter, fn_name)

    inp = torch.randn(M, 2 * N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)

    for _ in range(num_warmup):
        fn(out, inp)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    fn(out, inp)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_silu_and_mul(M, N, in_dtype="bf16", out_dtype="bf16",
                      num_warmup=3, **_):
    """``aiter.silu_and_mul`` (SwiGLU gate*up).

    Kernel supports BF16 / FP16 input and BF16 / FP16 / FP32 output (with
    upcast accumulation). Production deployments use BF16 in / BF16 out.
    """
    _activation_and_mul(
        "silu_and_mul", M, N,
        in_dtype=in_dtype, out_dtype=out_dtype, num_warmup=num_warmup,
    )


def test_gelu_and_mul(M, N, in_dtype="bf16", out_dtype="bf16",
                      num_warmup=3, **_):
    """``aiter.gelu_and_mul`` (GeGLU)."""
    _activation_and_mul(
        "gelu_and_mul", M, N,
        in_dtype=in_dtype, out_dtype=out_dtype, num_warmup=num_warmup,
    )


def test_gelu_tanh_and_mul(M, N, in_dtype="bf16", out_dtype="bf16",
                           num_warmup=3, **_):
    """``aiter.gelu_tanh_and_mul`` (tanh-approximated GeGLU)."""
    _activation_and_mul(
        "gelu_tanh_and_mul", M, N,
        in_dtype=in_dtype, out_dtype=out_dtype, num_warmup=num_warmup,
    )


# ---------------------------------------------------------------------------
# Quantization kernels
# ---------------------------------------------------------------------------

def test_dynamic_per_token_scaled_quant(
    M, N,
    in_dtype="bf16",
    out_dtype="fp8",
    scale_dtype="fp32",
    num_warmup=3,
    **_,
):
    """``aiter.dynamic_per_token_scaled_quant`` (per-token FP8 quantization).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
    out_dtype : {"fp8"}
        FP8 output. Storage selected by the active aiter dtype namespace
        (``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn`` on gfx950).
    scale_dtype : {"fp32"}
        Per-token scale dtype.
    """
    import torch
    import aiter

    in_t = _resolve_dtype(in_dtype)
    out_t = _resolve_dtype(out_dtype)
    scl_t = _resolve_dtype(scale_dtype)

    device = "cuda"
    print(
        f"test: dynamic_per_token_scaled_quant M={M} N={N} "
        f"in={in_dtype} out={out_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)
    scales = torch.empty(M, 1, dtype=scl_t, device=device)

    for _ in range(num_warmup):
        aiter.dynamic_per_token_scaled_quant(out, inp, scales)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    aiter.dynamic_per_token_scaled_quant(out, inp, scales)
    torch.cuda.synchronize()
    print(f"test: done, shape={out.shape}", flush=True)


def test_vllm_triton_group_quant_fp8(
    M, N,
    in_dtype="bf16",
    out_dtype="fp8",
    scale_dtype="fp32",
    group_size=128,
    num_warmup=3,
    **_,
):
    """``vllm::triton_per_token_group_quant_fp8`` (Triton FP8 group quant).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
    out_dtype : {"fp8"}
        FP8 output. Storage selected by the active aiter dtype namespace.
    scale_dtype : {"fp32"}
    group_size : int (default 128). Last-dim chunking for the per-group
        scales.
    """
    import torch
    from vllm.model_executor.layers.quantization.utils import fp8_utils  # noqa: F401

    in_t = _resolve_dtype(in_dtype)
    _resolve_dtype(out_dtype)  # validate

    device = "cuda"
    print(
        f"test: vllm_triton_group_quant_fp8 M={M} N={N} group_size={group_size} "
        f"in={in_dtype} out={out_dtype}",
        flush=True,
    )

    x = torch.randn(M, N, dtype=in_t, device=device)
    fn = torch.ops.vllm.triton_per_token_group_quant_fp8

    for _ in range(num_warmup):
        x_q, scales = fn(x, group_size)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    x_q, scales = fn(x, group_size)
    torch.cuda.synchronize()
    print(f"test: done, x_q={x_q.shape} scales={scales.shape}", flush=True)


# ---------------------------------------------------------------------------
# fused_flatten_mxfp4_quant  — SGLang triton fused flatten + MXFP4 quant
# ---------------------------------------------------------------------------

def test_fused_flatten_mxfp4_quant(
    M, N,
    in_dtype="bf16",
    group_size=128,
    num_warmup=3,
    **_,
):
    """SGLang triton fused flatten + MXFP4 quantization kernel.

    Accepts a 3-D input ``[M, N // group_size, group_size]`` (or equivalently
    the reshaped 2-D ``[M, N]``), quantizes each group to 4-bit MXFP4, and
    emits FP4x2-packed output plus per-group FP32 scales.

    Parameters
    ----------
    M : int
        Token count.
    N : int
        Hidden dimension (must be divisible by group_size).
    in_dtype : {"bf16", "fp16"}
    group_size : int (default 128)
    """
    import torch
    in_t = _resolve_dtype(in_dtype)
    device = "cuda"

    if N % group_size != 0:
        raise ValueError(f"N ({N}) must be divisible by group_size ({group_size})")

    N1 = N // group_size
    N2 = group_size

    print(
        f"test: fused_flatten_mxfp4_quant M={M} N={N} group_size={group_size} "
        f"in={in_dtype}",
        flush=True,
    )

    x = torch.randn(M, N1, N2, dtype=in_t, device=device)

    try:
        import sgl_kernel
        _fn = sgl_kernel.fused_mxfp4_quant
        def _call(): return _fn(x)  # noqa: E731
    except (ImportError, AttributeError):
        try:
            from aiter.triton import fused_flatten_mxfp4_quant as _triton_fn
            def _call(): return _triton_fn(x)  # noqa: E731
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Neither sgl_kernel.fused_mxfp4_quant nor "
                "aiter.triton.fused_flatten_mxfp4_quant is available."
            ) from exc

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print("test: done", flush=True)


# ---------------------------------------------------------------------------
# DSV3 / DSV4 group-quant harnesses
# ---------------------------------------------------------------------------

def test_dsv3_fused_flatten_fp8_group_quant(M, N, group_size=128, num_warmup=3, **_):
    """``aiter.ops.triton.quant.fused_fp8_quant.fused_flatten_fp8_group_quant``.

    Reshapes ``(M, N1, N2)`` to ``(M, N1*N2)`` and per-token-group FP8
    quantizes along the trailing dim with the given ``group_size``.

    We pass ``--M`` and ``--N`` as the input's leading and trailing dims, then
    pick ``N1 = N / group_size`` and ``N2 = group_size`` so the flattened
    output matches the DSV3 trace shape ``(32, 16, 128) -> (32, 2048)``.
    """
    import torch
    from aiter.ops.triton.quant.fused_fp8_quant import fused_flatten_fp8_group_quant

    device = "cuda"
    if N % group_size != 0:
        raise ValueError(f"N ({N}) must be divisible by group_size ({group_size})")
    N1 = N // group_size
    N2 = group_size
    print(
        f"test: dsv3_fused_flatten_fp8_group_quant M={M} N1={N1} N2={N2} group_size={group_size}",
        flush=True,
    )

    x = torch.randn(M, N1, N2, dtype=torch.bfloat16, device=device)

    for _ in range(num_warmup):
        out, scale = fused_flatten_fp8_group_quant(x, group_size)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    out, scale = fused_flatten_fp8_group_quant(x, group_size)
    torch.cuda.synchronize()
    print(f"test: done, out={out.shape} scale={scale.shape}", flush=True)


def test_dsv4_dynamic_per_group_scaled_quant(M=1819, N=7168, group_size=32, num_warmup=3, **_):
    """``aiter.dynamic_per_group_scaled_quant`` — per-group dynamic MX-FP8 quant."""
    import torch
    import aiter

    K = N
    print(f"test: dsv4_dynamic_per_group_scaled_quant M={M} K={K} group_size={group_size}", flush=True)
    input_t = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, K, device="cuda", dtype=aiter.dtypes.fp8)
    scales = torch.empty(M, K // group_size, device="cuda", dtype=aiter.dtypes.fp8_e8m0)

    for _ in range(num_warmup):
        aiter.dynamic_per_group_scaled_quant(out, input_t, scales,
                                             group_size=group_size, shuffle_scale=False)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    aiter.dynamic_per_group_scaled_quant(out, input_t, scales,
                                         group_size=group_size, shuffle_scale=False)
    torch.cuda.synchronize()
    print(f"test: done out={tuple(out.shape)} scales={tuple(scales.shape)}", flush=True)


# ---------------------------------------------------------------------------
# DSV4 mHC (multi-residual hyper-connection) family
# ---------------------------------------------------------------------------

def test_dsv4_mhc_pre_gemm_sqrsum(M=1819, N=7168, hc_mult=4, num_warmup=3, **_):
    """``aiter.mhc_pre_gemm_sqrsum`` — flattened-residual GEMM + sum-of-squares."""
    import torch
    import aiter
    from aiter.ops.mhc import mhc_pre_gemm_sqrsum, get_mhc_pre_splitk

    C = N
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
    hc_hidden = hc_mult * C                       # 28672
    print(f"test: dsv4_mhc_pre_gemm_sqrsum M={M} hc_mult={hc_mult} C={C}", flush=True)

    residual = torch.randn(M, hc_mult, C, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(hc_mult3, hc_hidden, dtype=torch.float32, device="cuda")
    split_k, tile_k = get_mhc_pre_splitk(M, hc_hidden)

    out_pad = torch.empty(split_k, M, (hc_mult3 + 31) // 32 * 32,
                          dtype=torch.float32, device="cuda")
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(split_k, M, dtype=torch.float32, device="cuda")

    for _ in range(num_warmup):
        mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, tile_k)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, tile_k)
    torch.cuda.synchronize()
    print(f"test: done split_k={split_k} tile_k={tile_k}", flush=True)


def test_dsv4_mhc_pre_big_fuse(M=1819, N=7168, hc_mult=4, split_k=16, num_warmup=3, **_):
    """``aiter.mhc_pre_big_fuse`` — RMS-norm + Sinkhorn + fused residual mix."""
    import torch
    from aiter.ops.mhc import mhc_pre_big_fuse, get_mhc_pre_splitk

    C = N
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    try:
        split_k, _ = get_mhc_pre_splitk(M, hc_mult * C)
    except Exception:
        pass
    print(f"test: dsv4_mhc_pre_big_fuse M={M} hc_mult={hc_mult} C={C} split_k={split_k}", flush=True)

    residual = torch.randn(M, hc_mult, C, dtype=torch.bfloat16, device="cuda")
    gemm_out_mul = torch.randn(split_k, M, hc_mult3, dtype=torch.float32, device="cuda")
    gemm_out_sqrsum = torch.rand(split_k, M, dtype=torch.float32, device="cuda") + 0.1
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda") * 0.1

    post_mix = torch.empty(M, hc_mult, 1, dtype=torch.float32, device="cuda")
    comb_mix = torch.empty(M, hc_mult, hc_mult, dtype=torch.float32, device="cuda")
    layer_input = torch.empty(M, C, dtype=torch.bfloat16, device="cuda")

    def _call():
        mhc_pre_big_fuse(
            post_mix, comb_mix, layer_input,
            gemm_out_mul, gemm_out_sqrsum,
            hc_scale, hc_base, residual,
            rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0, sinkhorn_repeat=20,
        )

    for _ in range(num_warmup):
        _call()
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    _call()
    torch.cuda.synchronize()
    print("test: done", flush=True)


def test_dsv4_mhc_post(M=1819, N=7168, hc_mult=4, num_warmup=3, **_):
    """``aiter.mhc_post`` — merge block output back into n residual streams."""
    import torch
    from aiter.ops.mhc import mhc_post

    C = N
    print(f"test: dsv4_mhc_post M={M} hc_mult={hc_mult} C={C}", flush=True)

    x = torch.randn(M, C, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(M, hc_mult, C, dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn(M, hc_mult, 1, dtype=torch.float32, device="cuda")
    comb_res_mix = torch.randn(M, hc_mult, hc_mult, dtype=torch.float32, device="cuda")
    out = torch.empty_like(residual)

    for _ in range(num_warmup):
        mhc_post(out, x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    print("test: measured iteration...", flush=True)
    mhc_post(out, x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    print(f"test: done out={tuple(out.shape)}", flush=True)


# ---------------------------------------------------------------------------
# OP_METADATA
# ---------------------------------------------------------------------------

OP_METADATA: dict = {
    "silu_and_mul": {
        "fn":           test_silu_and_mul,
        "category":     "UnaryElementwise",
        "description":  "AITER silu_and_mul (SwiGLU gate*up, BF16)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "gelu_and_mul": {
        "fn":           test_gelu_and_mul,
        "category":     "UnaryElementwise",
        "description":  "AITER gelu_and_mul (GeGLU)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "gelu_tanh_and_mul": {
        "fn":           test_gelu_tanh_and_mul,
        "category":     "UnaryElementwise",
        "description":  "AITER gelu_tanh_and_mul (tanh-approximated GeGLU)",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "dynamic_per_token_scaled_quant": {
        "fn":           test_dynamic_per_token_scaled_quant,
        "category":     "GroupQuant",
        "description":  "AITER dynamic per-token FP8 quantization",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "in_dtype": "bf16", "out_dtype": "fp8"},
        "required_args": ["M", "N"],
    },
    "vllm_triton_group_quant_fp8": {
        "fn":           test_vllm_triton_group_quant_fp8,
        "category":     "GroupQuant",
        "description":  "vLLM Triton per-token group FP8 quantization",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 2048, "N": 4096, "in_dtype": "bf16", "out_dtype": "fp8"},
        "required_args": ["M", "N"],
    },
    "fused_flatten_mxfp4_quant": {
        "fn":           test_fused_flatten_mxfp4_quant,
        "category":     "GroupQuant",
        "description":  "SGLang triton fused flatten + MXFP4 quant",
        "dtypes":       ["bf16", "fp16"],
        "defaults":     {"M": 822, "N": 7168, "group_size": 128, "in_dtype": "bf16"},
        "required_args": ["M", "N"],
    },
    "dsv3_fused_flatten_fp8_group_quant": {
        "fn":           test_dsv3_fused_flatten_fp8_group_quant,
        "category":     "GroupQuant",
        "description":  "DSV3 AITER triton fused flatten + FP8 group quant",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 32, "N": 2048, "group_size": 128},
        "required_args": ["M", "N"],
    },
    "dsv4_dynamic_per_group_scaled_quant": {
        "fn":           test_dsv4_dynamic_per_group_scaled_quant,
        "category":     "GroupQuant",
        "description":  "DSV4 per-group dynamic MX-FP8 quant (aiter)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1819, "N": 7168, "group_size": 32},
        "required_args": ["M", "N"],
    },
    "dsv4_mhc_pre_gemm_sqrsum": {
        "fn":           test_dsv4_mhc_pre_gemm_sqrsum,
        "category":     "mHC_pre",
        "description":  "DSV4 mHC pre GEMM + sum-of-squares (aiter)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1819, "N": 7168},
        "required_args": ["M", "N"],
    },
    "dsv4_mhc_pre_big_fuse": {
        "fn":           test_dsv4_mhc_pre_big_fuse,
        "category":     "mHC_pre",
        "description":  "DSV4 mHC pre RMS+Sinkhorn+mix fuse (aiter)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1819, "N": 7168},
        "required_args": ["M", "N"],
    },
    "dsv4_mhc_post": {
        "fn":           test_dsv4_mhc_post,
        "category":     "mHC_post",
        "description":  "DSV4 mHC post stream-merge (aiter)",
        "dtypes":       ["bf16"],
        "defaults":     {"M": 1819, "N": 7168},
        "required_args": ["M", "N"],
    },
}
