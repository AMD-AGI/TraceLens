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
}
