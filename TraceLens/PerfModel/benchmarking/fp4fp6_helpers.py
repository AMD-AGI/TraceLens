###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Triton block-scaled MXFP4/MXFP6 GEMM microbenchmarks.

Issues ``tl.dot_scaled`` on packed-uint8 operands with uint8 e8m0 per-block
scales. On MI350 this lowers to ``v_mfma_scale_f32_*_f8f6f4``; on MI300 to the
emulated software path. Probes ``e2m1``/``e3m2``/``e2m3`` at import; falls
back to ``0.0`` if unsupported.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

# OCP MX scaling block size: 32 elements share one E8M0 scale byte. Triton does
# not expose this as a Python constant; tl.dot_scaled assumes group_size=32 for
# e8m0 scales (https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html).
# OCP Microscaling Formats (MX) v1.0, Section 5.2:
# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
MX_BLOCK = 32


def triton_available() -> bool:
    return triton is not None and tl is not None


def _random_packed_e2m1(rows: int, k: int, device: torch.device) -> torch.Tensor:
    """Random uint8 tensor of shape (rows, k // 2), two E2M1 nibbles per byte."""
    lo = torch.randint(0, 16, (rows, k // 2), dtype=torch.uint8, device=device)
    hi = torch.randint(0, 16, (rows, k // 2), dtype=torch.uint8, device=device)
    return (hi << 4) | lo


def _random_e8m0_scales(rows: int, nb: int, device: torch.device) -> torch.Tensor:
    """Random uint8 e8m0 scales near unity (bias 127, so 124..127 ~ [0.125, 1.0])."""
    return torch.randint(124, 128, (rows, nb), dtype=torch.uint8, device=device)


def prepare_mxfp4_gemm(
    m: int, n: int, k: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (a_packed, b_packed, a_scales, b_scales, c) for MXFP4 (2 nibbles/byte)."""
    assert k % MX_BLOCK == 0 and k % 2 == 0
    nb = k // MX_BLOCK
    a_packed = _random_packed_e2m1(m, k, device)
    b_packed = _random_packed_e2m1(n, k, device)
    a_scales = _random_e8m0_scales(m, nb, device)
    b_scales = _random_e8m0_scales(n, nb, device)
    c = torch.empty(m, n, device=device, dtype=torch.float16)
    return a_packed, b_packed, a_scales, b_scales, c


def prepare_mxfp6_gemm(
    m: int, n: int, k: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build MXFP6 inputs in 1-byte-per-value (FP8-stride) layout."""
    assert k % MX_BLOCK == 0
    nb = k // MX_BLOCK
    a_packed = torch.randint(0, 256, (m, k), dtype=torch.uint8, device=device)
    b_packed = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=device)
    a_scales = _random_e8m0_scales(m, nb, device)
    b_scales = _random_e8m0_scales(n, nb, device)
    c = torch.empty(m, n, device=device, dtype=torch.float16)
    return a_packed, b_packed, a_scales, b_scales, c


if triton_available():

    @triton.jit
    def _mx_dot_scaled_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scales_ptr,
        b_scales_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LHS_DTYPE: tl.constexpr,
        RHS_DTYPE: tl.constexpr,
        PACK: tl.constexpr,  # values per byte along K (2 for fp4, 1 for fp8)
    ):
        SCALE_GROUP_SIZE: tl.constexpr = 32  # must match MX_BLOCK / tl.dot_scaled e8m0 group size
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # Packed-K offsets: each uint8 holds PACK codes.
        offs_k = tl.arange(0, BLOCK_K // PACK)
        offs_sk = tl.arange(0, BLOCK_K // SCALE_GROUP_SIZE)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_scale_ptrs = (
            a_scales_ptr + offs_m[:, None] * stride_asm + offs_sk[None, :] * stride_ask
        )
        b_scale_ptrs = (
            b_scales_ptr + offs_n[:, None] * stride_bsn + offs_sk[None, :] * stride_bsk
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            a_scales = tl.load(a_scale_ptrs)
            b_scales = tl.load(b_scale_ptrs)
            acc = tl.dot_scaled(a, a_scales, LHS_DTYPE, b, b_scales, RHS_DTYPE, acc)

            a_ptrs += (BLOCK_K // PACK) * stride_ak
            b_ptrs += (BLOCK_K // PACK) * stride_bk
            a_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_K // SCALE_GROUP_SIZE) * stride_bsk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.type.element_ty))


def _probe_dot_scaled(lhs_dtype: str, rhs_dtype: str, pack: int) -> bool:
    """Compile a tiny kernel to confirm tl.dot_scaled accepts these dtypes."""
    if not triton_available() or not torch.cuda.is_available():
        return False
    try:
        dev = torch.device("cuda:0")
        m = n = 64
        k = 64
        if pack == 2:
            a, b, sa, sb, c = prepare_mxfp4_gemm(m, n, k, dev)
        else:
            a, b, sa, sb, c = prepare_mxfp6_gemm(m, n, k, dev)
        w = b.T.contiguous()
        _mx_dot_scaled_kernel[(1, 1)](
            a,
            w,
            c,
            sa,
            sb,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            w.stride(0),
            w.stride(1),
            c.stride(0),
            c.stride(1),
            sa.stride(0),
            sa.stride(1),
            sb.stride(0),
            sb.stride(1),
            BLOCK_M=m,
            BLOCK_N=n,
            BLOCK_K=k,
            LHS_DTYPE=lhs_dtype,
            RHS_DTYPE=rhs_dtype,
            PACK=pack,
            num_warps=4,
        )
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


_MXFP4_SUPPORTED: bool = False
# _MXFP6_KIND: "native" (e3m2/e2m3), "f6f4_via_e2m1" (fallback, same MFMA), or "".
_MXFP6_DTYPE: str = ""
_MXFP6_PACK: int = 0
_MXFP6_KIND: str = ""


def _resolve_support() -> None:
    global _MXFP4_SUPPORTED, _MXFP6_DTYPE, _MXFP6_PACK, _MXFP6_KIND
    _MXFP4_SUPPORTED = _probe_dot_scaled("e2m1", "e2m1", pack=2)

    for dt in ("e3m2", "e2m3"):
        if _probe_dot_scaled(dt, dt, pack=2):
            _MXFP6_DTYPE = dt
            _MXFP6_PACK = 2
            _MXFP6_KIND = "native"
            print(f"[fp4fp6] MXFP6: native '{dt}'")
            return

    if _MXFP4_SUPPORTED:
        _MXFP6_DTYPE = "e2m1"
        _MXFP6_PACK = 2
        _MXFP6_KIND = "f6f4_via_e2m1"
        print("[fp4fp6] MXFP6: e2m1 fallback (same f8f6f4 MFMA, HW-peak TFLOPS).")
    else:
        print("[fp4fp6] MXFP6: unsupported; will return 0.")


if triton_available() and torch.cuda.is_available():
    try:
        _resolve_support()
    except Exception:  # pragma: no cover
        pass


def _launch_scaled_gemm(
    a: torch.Tensor,
    b_nk: torch.Tensor,  # stored as (N, K // PACK)
    sa: torch.Tensor,
    sb: torch.Tensor,
    c: torch.Tensor,
    lhs_dtype: str,
    rhs_dtype: str,
    pack: int,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 256,
) -> None:
    m, packed_k = a.shape
    k = packed_k * pack
    n = b_nk.shape[0]
    w = b_nk.T  # kernel expects (K // PACK, N)

    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _mx_dot_scaled_kernel[grid](
        a,
        w,
        c,
        sa,
        sb,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(0),
        c.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        LHS_DTYPE=lhs_dtype,
        RHS_DTYPE=rhs_dtype,
        PACK=pack,
        num_warps=8,
    )


def bench_mxfp4_gemm(
    m: int,
    n: int,
    k: int,
    device: int,
    *,
    warmup: int,
    rep: int,
    do_bench_fn,
) -> float:
    if not triton_available() or not _MXFP4_SUPPORTED:
        return 0.0
    if k % MX_BLOCK != 0 or k % 2 != 0:
        return 0.0
    dev = torch.device(f"cuda:{device}")
    a, b, sa, sb, c = prepare_mxfp4_gemm(m, n, k, dev)

    def _run() -> None:
        _launch_scaled_gemm(a, b, sa, sb, c, "e2m1", "e2m1", pack=2)

    ms = do_bench_fn(_run, warmup=warmup, rep=rep)
    return (2 * m * n * k) / (ms * 1e-3) / 1e12


_AITER_MXFP4_CACHE: dict = {
    "checked": False,
    "ok": False,
    "quant": None,
    "shuffle": None,
}
_AITER_INT8_CACHE: dict = {"checked": False, "ok": False, "fn": None}


def _aiter_mxfp4_ready() -> bool:
    """Lazy-import aiter and confirm the gfx950 CK MXFP4 path is usable."""
    if _AITER_MXFP4_CACHE["checked"]:
        return _AITER_MXFP4_CACHE["ok"]
    _AITER_MXFP4_CACHE["checked"] = True
    try:
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        try:
            from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
        except ImportError:
            from aiter.jit.utils.chip_info import get_gfx

        if get_gfx() not in ("gfx950",):
            return False
        quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        _AITER_MXFP4_CACHE["quant"] = quant
        _AITER_MXFP4_CACHE["shuffle"] = shuffle_weight
        _AITER_MXFP4_CACHE["gemm"] = aiter.gemm_a4w4
        _AITER_MXFP4_CACHE["ok"] = True
        return True
    except Exception:
        return False


def bench_mxfp4_ck_gemm(
    m: int,
    n: int,
    k: int,
    device: int,
    *,
    warmup: int,
    rep: int,
    do_bench_fn,
) -> float:
    """Bench aiter's gfx950 CK ``gemm_a4w4`` (block-scaled MXFP4)."""
    if not _aiter_mxfp4_ready():
        return 0.0
    if k % MX_BLOCK != 0 or k % 2 != 0:
        return 0.0
    dev = torch.device(f"cuda:{device}")
    quant = _AITER_MXFP4_CACHE["quant"]
    shuffle_weight = _AITER_MXFP4_CACHE["shuffle"]
    gemm_a4w4 = _AITER_MXFP4_CACHE["gemm"]
    try:
        x_fp = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
        w_fp = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        x_q, x_scales = quant(x_fp, shuffle=True)
        w_q, w_scales = quant(w_fp, shuffle=True)
        wshuffle = shuffle_weight(w_q, layout=(16, 16))
    except Exception:
        return 0.0

    def _run() -> None:
        gemm_a4w4(x_q, wshuffle, x_scales, w_scales, bpreshuffle=True)

    try:
        ms = do_bench_fn(_run, warmup=warmup, rep=rep)
    except Exception:
        return 0.0
    return (2 * m * n * k) / (ms * 1e-3) / 1e12


def _aiter_int8_ready() -> bool:
    """Lazy-import aiter and confirm its CK INT8 GEMM (gemm_a8w8) is usable."""
    if _AITER_INT8_CACHE.get("checked"):
        return bool(_AITER_INT8_CACHE.get("ok"))
    _AITER_INT8_CACHE["checked"] = True
    try:
        import aiter
    except Exception as e:
        _AITER_INT8_CACHE["error"] = f"aiter import failed: {e}"
        print(f"[fp4fp6] INT8 CK: {_AITER_INT8_CACHE['error']}", flush=True)
        return False
    if not hasattr(aiter, "gemm_a8w8"):
        _AITER_INT8_CACHE["error"] = "aiter.gemm_a8w8 attribute missing"
        print(f"[fp4fp6] INT8 CK: {_AITER_INT8_CACHE['error']}", flush=True)
        return False
    _AITER_INT8_CACHE["fn"] = aiter.gemm_a8w8
    _AITER_INT8_CACHE["ok"] = True
    return True


def bench_int8_ck_gemm(
    m: int,
    n: int,
    k: int,
    device: int,
    *,
    warmup: int,
    rep: int,
    do_bench_fn,
) -> float:
    """Bench aiter's CK ``gemm_a8w8`` (per-tensor scaled INT8 -> bf16)."""
    if not _aiter_int8_ready():
        return 0.0
    dev = torch.device(f"cuda:{device}")
    fn = _AITER_INT8_CACHE["fn"]
    try:
        XQ = torch.randint(-128, 127, (m, k), dtype=torch.int8, device=dev)
        WQ = torch.randint(-128, 127, (n, k), dtype=torch.int8, device=dev)
        x_scale = torch.ones((m, 1), dtype=torch.float32, device=dev)
        w_scale = torch.ones((1, n), dtype=torch.float32, device=dev)
        out_dtype = torch.bfloat16
        bias = torch.zeros((1, n), dtype=out_dtype, device=dev)
    except Exception as e:
        print(f"[fp4fp6] INT8 CK: input alloc failed @ ({m},{n},{k}): {e}", flush=True)
        return 0.0

    def _run() -> None:
        fn(XQ, WQ, x_scale, w_scale, bias, dtype=out_dtype)

    # Warm once outside do_bench to surface compile/dispatch errors clearly.
    try:
        _run()
        torch.cuda.synchronize()
    except Exception as e:
        print(
            f"[fp4fp6] INT8 CK: gemm_a8w8 failed @ ({m},{n},{k}): {e}",
            flush=True,
        )
        return 0.0

    try:
        ms = do_bench_fn(_run, warmup=warmup, rep=rep)
    except Exception as e:
        print(
            f"[fp4fp6] INT8 CK: do_bench failed @ ({m},{n},{k}): {e}",
            flush=True,
        )
        return 0.0
    return (2 * m * n * k) / (ms * 1e-3) / 1e12


def bench_mxfp6_gemm(
    m: int,
    n: int,
    k: int,
    device: int,
    *,
    warmup: int,
    rep: int,
    do_bench_fn,
) -> float:
    if not triton_available() or not _MXFP6_DTYPE:
        return 0.0
    if k % MX_BLOCK != 0:
        return 0.0
    if _MXFP6_PACK == 2 and k % 2 != 0:
        return 0.0
    dev = torch.device(f"cuda:{device}")
    if _MXFP6_PACK == 2:
        a, b, sa, sb, c = prepare_mxfp4_gemm(m, n, k, dev)
    else:
        a, b, sa, sb, c = prepare_mxfp6_gemm(m, n, k, dev)

    def _run() -> None:
        _launch_scaled_gemm(
            a, b, sa, sb, c, _MXFP6_DTYPE, _MXFP6_DTYPE, pack=_MXFP6_PACK
        )

    ms = do_bench_fn(_run, warmup=warmup, rep=rep)
    return (2 * m * n * k) / (ms * 1e-3) / 1e12
