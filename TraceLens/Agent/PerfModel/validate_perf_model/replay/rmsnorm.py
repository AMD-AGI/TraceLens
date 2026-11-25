###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RMSNorm-family test harnesses for ``validate_perf_model``.

Each ``test_<op>`` function builds the input tensors and returns the kernel
call, which ``replay/_runner.py`` measures. Functions are parameterized over input/weight/output dtype
where the kernel supports it; quant variants additionally honor
``out_dtype`` to select between FP8 storage flavors.
"""

# NOTE: torch is imported lazily inside each function so that this module can
# be imported by the parent process for argv-building without paying the cost
# (and side effects) of importing torch.

from ._dtypes import resolve_dtype as _resolve_dtype


def test_rms_norm(M, N, in_dtype="bf16", w_dtype="bf16", out_dtype="bf16", **_):
    """``aiter.rms_norm`` (CK).

    Parameters
    ----------
    in_dtype, w_dtype, out_dtype : {"bf16", "fp16"}
        Activation, weight, and output dtypes. The kernel requires all three
        to match.
    """
    import torch
    import aiter

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    if not (in_t == w_t == out_t):
        raise ValueError(
            f"rms_norm requires matching dtypes; got "
            f"in={in_dtype} w={w_dtype} out={out_dtype}"
        )

    device = "cuda"
    print(f"test: rms_norm M={M} N={N} dtype={in_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)

    def call():
        return aiter.rms_norm(inp, weight, 1e-06)

    return call


def test_rmsnorm(M, N, in_dtype="bf16", w_dtype="bf16", out_dtype="bf16", **_):
    """``aiter.ops.rmsnorm.rmsnorm`` (compiled CK op, out-first API).

    Same dtype constraints as :func:`test_rms_norm` -- input/weight/output
    must match.
    """
    import torch
    from aiter.ops.rmsnorm import rmsnorm

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    if not (in_t == w_t == out_t):
        raise ValueError(
            f"rmsnorm requires matching dtypes; got "
            f"in={in_dtype} w={w_dtype} out={out_dtype}"
        )

    device = "cuda"
    print(f"test: rmsnorm M={M} N={N} dtype={in_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)
    out = torch.empty_like(inp)

    def call():
        rmsnorm(out, inp, weight, 1e-06)

    return call


def test_add_rmsnorm(M, N, in_dtype="bf16", w_dtype="bf16", out_dtype="bf16", **_):
    """``aiter.fused_add_rms_norm_cu`` (residual-add + RMSNorm).

    Parameters mirror :func:`test_rms_norm`. The kernel updates ``inp`` and
    ``residual`` in-place and requires all three dtypes to match.
    """
    import torch
    import aiter

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    if not (in_t == w_t == out_t):
        raise ValueError(
            f"add_rmsnorm requires matching dtypes; got "
            f"in={in_dtype} w={w_dtype} out={out_dtype}"
        )

    device = "cuda"
    print(f"test: add_rmsnorm M={M} N={N} dtype={in_dtype}", flush=True)

    inp = torch.randn(M, N, dtype=in_t, device=device)
    residual = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)

    def call():
        aiter.fused_add_rms_norm_cu(inp.clone(), residual.clone(), weight, 1e-06)

    return call


def test_rmsnorm_dynamicquant(
    M,
    N,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="fp8",
    scale_dtype="fp32",
    **_,
):
    """``aiter.ops.rmsnorm.rmsnorm2d_fwd_with_dynamicquant`` (fused RMSNorm + dynamic FP8 quant).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
    w_dtype : {"bf16", "fp16"}
    out_dtype : {"fp8"}
        FP8 output. Storage selected by the active aiter dtype namespace
        (``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn`` on gfx950).
    scale_dtype : {"fp32"}
        Per-token scale dtype.
    """
    import torch
    from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_dynamicquant

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    out_t = _resolve_dtype(out_dtype)
    scl_t = _resolve_dtype(scale_dtype)

    device = "cuda"
    print(
        f"test: rmsnorm_dynamicquant M={M} N={N} in={in_dtype} out={out_dtype}",
        flush=True,
    )

    inp = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)
    out = torch.empty(M, N, dtype=out_t, device=device)
    yscale = torch.empty(M, 1, dtype=scl_t, device=device)

    def call():
        rmsnorm2d_fwd_with_dynamicquant(out, inp, yscale, weight, 1e-06)

    return call


def test_vllm_rmsnorm_fp8_group_quant(
    M,
    N,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="fp8",
    scale_dtype="fp32",
    group_size=128,
    **_,
):
    """``vllm::rocm_aiter_rmsnorm_fp8_group_quant`` (fused RMSNorm + FP8 group quant).

    Parameters
    ----------
    in_dtype : {"bf16", "fp16"}
    w_dtype : {"bf16", "fp16"}
    out_dtype : {"fp8"}
        FP8 output (group-quantized along the last dim with stride
        ``group_size``).
    scale_dtype : {"fp32"}
    group_size : int (default 128).
    """
    import torch
    import vllm._aiter_ops  # noqa: F401  registers torch.ops.vllm.rocm_aiter_*

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    _resolve_dtype(out_dtype)

    device = "cuda"
    eps = 1e-06
    print(
        f"test: vllm_rmsnorm_fp8_group_quant M={M} N={N} gs={group_size} "
        f"in={in_dtype} out={out_dtype}",
        flush=True,
    )

    x = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)
    fn = torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant

    def call():
        return fn(x, weight, eps, group_size)

    return call


def test_vllm_rmsnorm_add_fp8_group_quant(
    M,
    N,
    in_dtype="bf16",
    w_dtype="bf16",
    out_dtype="fp8",
    scale_dtype="fp32",
    group_size=128,
    **_,
):
    """``vllm::rocm_aiter_rmsnorm_with_add_fp8_group_quant``."""
    import torch
    import vllm._aiter_ops  # noqa: F401  registers torch.ops.vllm.rocm_aiter_*

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    _resolve_dtype(out_dtype)

    device = "cuda"
    eps = 1e-06
    print(
        f"test: vllm_rmsnorm_add_fp8_group_quant M={M} N={N} gs={group_size} "
        f"in={in_dtype} out={out_dtype}",
        flush=True,
    )

    x = torch.randn(M, N, dtype=in_t, device=device)
    residual = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.randn(N, dtype=w_t, device=device)
    fn = torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant

    def call():
        return fn(x.clone(), residual.clone(), weight, eps, group_size)

    return call


def test_fused_rms_mxfp4_quant(M, N, in_dtype="bf16", w_dtype="bf16", **_):
    """SGLang triton fused RMSNorm + MXFP4 quantization kernel.

    Single-path variant: normalizes ``x1 [M, N]`` with weight ``w [N]`` and
    immediately quantizes the output to MXFP4.  No residual (x2/res1) input.

    Parameters
    ----------
    M : int
        Token count.
    N : int
        Hidden dimension (= weight length).
    in_dtype, w_dtype : {"bf16", "fp16"}
    """
    import torch

    in_t = _resolve_dtype(in_dtype)
    w_t = _resolve_dtype(w_dtype)
    device = "cuda"

    print(
        f"test: fused_rms_mxfp4_quant M={M} N={N} dtype={in_dtype}",
        flush=True,
    )

    x = torch.randn(M, N, dtype=in_t, device=device)
    weight = torch.ones(N, dtype=w_t, device=device)

    try:
        import sgl_kernel

        _fn = sgl_kernel.fused_rms_mxfp4_quant

        def _call():
            return _fn(x, weight)  # noqa: E731

    except (ImportError, AttributeError):
        try:
            from aiter.triton import fused_rms_mxfp4_quant as _triton_fn

            def _call():
                return _triton_fn(x, weight)  # noqa: E731

        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Neither sgl_kernel.fused_rms_mxfp4_quant nor "
                "aiter.triton.fused_rms_mxfp4_quant is available."
            ) from exc

    return _call
