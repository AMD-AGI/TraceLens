#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU Microbenchmarking Suite.

Measures matrix TFLOPS (PyTorch GEMM), vector TFLOPS (Triton FMA chain), and
HBM bandwidth. Writes JSON in the ``results/MI300X.json`` shape. Methodology:
do_bench, L2 clear, warmup=30 rep=200, normal-distributed inputs, median ms.

Examples:
    # Default run on device 0; writes gpu_microbench_results.json
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0

    # Custom output path (parent dirs auto-created)
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0 \\
        --output results/mi355x.json

    # Pin to a specific physical GPU
    HIP_VISIBLE_DEVICES=2 python -m TraceLens.PerfModel.benchmarking.microbench \\
        --device 0 --output runs/card2.json

    # Faster smoke test (lower warmup/rep)
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0 --warmup 5 --rep 20

    # Skip vector / bandwidth sections
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0 \\
        --skip-vector --skip-bandwidth

    # Override idle check (run even if the GPU shows activity)
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0 --allow-busy

    # Large GEMM + multi-GB HBM sweep
    python -m TraceLens.PerfModel.benchmarking.microbench --device 0 \\
        --shape-sweep --sweep-output sweeps/mi355x.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from triton.testing import do_bench

from .microbench_utils import check_gpu_idle

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None

try:
    from .fp4fp6_helpers import (
        MX_BLOCK,
        bench_int8_ck_gemm,
        bench_mxfp4_ck_gemm,
        bench_mxfp4_gemm,
        bench_mxfp6_gemm,
        mx_available,
    )
except Exception:  # pragma: no cover
    MX_BLOCK = 32

    def mx_available() -> bool:
        return False

    def bench_mxfp4_gemm(*_a, **_k):
        return 0.0

    def bench_mxfp4_ck_gemm(*_a, **_k):
        return 0.0

    def bench_mxfp6_gemm(*_a, **_k):
        return 0.0

    def bench_int8_ck_gemm(*_a, **_k):
        return 0.0


WARMUP = 30
REP = 200

# Resolved once per process by :func:`_resolve_fp8_dtype`.
_FP8_DTYPE_CACHE: Optional[torch.dtype] = None

_FP8_DTYPE_CANDIDATES: Tuple[str, ...] = (
    "float8_e4m3fnuz",  # AMD MI300/MI35x MAF dtype when hipBLASLt supports it
    "float8_e4m3fn",
    "float8_e5m2",  # Often works on ROCm 7.2+ when e4m3fnuz has no solution
    "float8_e5m2fnuz",
)

# Same keys as `results/MI300X.json` (TraceLens arch template).
ARCH_TLOPS_KEYS: Tuple[str, ...] = (
    "matrix_fp16",
    "matrix_bf16",
    "matrix_fp32",
    "matrix_fp64",
    "matrix_fp8",
    "matrix_fp4",
    "matrix_fp6",
    "matrix_int8",
    "vector_fp16",
    "vector_bf16",
    "vector_fp32",
    "vector_fp64",
)

# GEMM shapes (M,N,K) 
GEMM_SHAPES: List[Tuple[int, int, int]] = [
    (8192, 8192, 8192),
    (16384, 8192, 1280),
    (16384, 1024, 8192),
    (16384, 8192, 7168),
    (16384, 3584, 8192),
]

# Sizes for memory bandwidth sweeps (bytes)
BW_SIZES: List[int] = [
    64 * 1024 * 1024,  # 64 MB
    128 * 1024 * 1024,  # 128 MB
    256 * 1024 * 1024,  # 256 MB
    512 * 1024 * 1024,  # 512 MB
    1024 * 1024 * 1024,  # 1 GB
    2 * 1024 * 1024 * 1024,  # 2 GB
]

# Extended sweep candidates (see --shape-sweep): large GEMM peaks only.
GEMM_SHAPES_SWEEP_LARGE: List[Tuple[int, int, int]] = [
    (12288, 12288, 12288),
    (16384, 16384, 16384),
    (16384, 16384, 8192),
    (16384, 8192, 16384),
    (20480, 8192, 8192),
    (24576, 8192, 8192),
]

# hipBLASLt / CDNA “sweet” shapes (many dims are multiples of 304).
GEMM_SHAPES_SWEEP_TILE304: List[Tuple[int, int, int]] = [
    (4864, 4096, 8256),
    (4864, 8192, 4160),
    (8192, 4864, 6878),
]

# Extended HBM sweep: multi-GB transfers to approach device HBM ceiling.
BW_SIZES_SWEEP_LARGE: List[int] = [
    4 * 1024 * 1024 * 1024,
    8 * 1024 * 1024 * 1024,
    16 * 1024 * 1024 * 1024,
    32 * 1024 * 1024 * 1024,
]


def _bpe(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _arch_product_name(gpu_name: str, mem_gb: float) -> str:
    """Short name to match arch JSONs like `MI300X.json` (e.g. 'MI300X')."""
    # ROCm containers often report a generic device string; use memory tier as hint.
    mem = int(round(mem_gb))
    if mem >= 280:
        return "MI355X"
    if mem >= 180:
        return "MI300X"
    parts = gpu_name.strip().split()
    return parts[-1] if parts else "GPU"


def _build_measured_arch_json(
    gpu_name: str,
    mem_gb: float,
    read_bw_gbps: float,
    matrix_results: Dict[str, float],
    vector_results: Dict[str, float],
) -> Dict:
    """
    One object shaped like `MI300X.json`: only `name`, `mem_bw_gbps`, `memory_gb`,
    and `max_achievable_tflops` with the canonical TFLOPS keys (no extras).
    """
    maf: Dict[str, float] = {}
    for key in ARCH_TLOPS_KEYS:
        if key.startswith("matrix_"):
            maf[key] = int(round(float(matrix_results.get(key, 0.0))))
        else:
            maf[key] = int(round(float(vector_results.get(key, 0.0))))
    return {
        "name": _arch_product_name(gpu_name, mem_gb),
        "mem_bw_gbps": int(round(float(read_bw_gbps))),
        "memory_gb": int(round(mem_gb)),
        "max_achievable_tflops": maf,
    }


# ── Matrix TFLOPS benchmarks ─────────────────────────────────────────


def _gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def bench_gemm(M: int, N: int, K: int, dtype: torch.dtype, device: int = 0) -> float:
    """Returns achieved TFLOPS for a single GEMM shape."""
    dev = f"cuda:{device}"
    A = torch.randn(M, K, dtype=dtype, device=dev)
    B = torch.randn(K, N, dtype=dtype, device=dev)

    ms = do_bench(lambda: torch.matmul(A, B), warmup=WARMUP, rep=REP)
    tflops = _gemm_flops(M, N, K) / (ms * 1e-3) / 1e12
    return tflops


def _fp8_dtypes_to_try() -> List[torch.dtype]:
    dtypes: List[torch.dtype] = []
    for name in _FP8_DTYPE_CANDIDATES:
        dt = getattr(torch, name, None)
        if dt is not None:
            dtypes.append(dt)
    return dtypes


def _resolve_fp8_dtype(device: int, M: int, N: int, K: int) -> Optional[torch.dtype]:
    """Pick the first FP8 dtype that runs ``torch._scaled_mm`` on this GPU/stack."""
    global _FP8_DTYPE_CACHE
    if _FP8_DTYPE_CACHE is not None:
        return _FP8_DTYPE_CACHE

    dev = f"cuda:{device}"
    scale_a = torch.ones(1, dtype=torch.float32, device=dev)
    scale_b = torch.ones(1, dtype=torch.float32, device=dev)
    probe_m = min(512, M)
    probe_n = min(512, N)
    probe_k = min(512, K)

    for fp8_dtype in _fp8_dtypes_to_try():
        try:
            a = torch.randn(probe_m, probe_k, device=dev).to(fp8_dtype)
            b = torch.randn(probe_n, probe_k, device=dev).to(fp8_dtype)
            torch._scaled_mm(
                a,
                b.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            )
            _FP8_DTYPE_CACHE = fp8_dtype
            print(f"    FP8 dtype: {fp8_dtype}")
            return fp8_dtype
        except Exception:
            continue

    print("    No supported FP8 dtype for torch._scaled_mm on this stack.")
    return None


def bench_gemm_fp8(M: int, N: int, K: int, device: int = 0) -> float:
    """FP8 GEMM via ``torch._scaled_mm`` (dtype auto-selected per GPU/stack)."""
    fp8_dtype = _resolve_fp8_dtype(device, M, N, K)
    if fp8_dtype is None:
        return 0.0

    dev = f"cuda:{device}"
    a = torch.randn(M, K, device=dev).to(fp8_dtype)
    b = torch.randn(N, K, device=dev).to(fp8_dtype)
    scale_a = torch.ones(1, dtype=torch.float32, device=dev)
    scale_b = torch.ones(1, dtype=torch.float32, device=dev)

    try:
        ms = do_bench(
            lambda: torch._scaled_mm(
                a,
                b.t(),
                scale_a=scale_a,
                scale_b=scale_b,
                out_dtype=torch.bfloat16,
            ),
            warmup=WARMUP,
            rep=REP,
        )
    except Exception as e:
        print(f"    FP8 scaled_mm failed ({e})")
        return 0.0

    return _gemm_flops(M, N, K) / (ms * 1e-3) / 1e12


def bench_gemm_int8(M: int, N: int, K: int, device: int = 0) -> float:
    """
    INT8 GEMM via ``torch._int_mm`` (available on ROCm and CUDA). TFLOPS:
    ``(2*M*N*K) / time / 1e12`` — same op count as :func:`_gemm_flops`.
    """
    dev = f"cuda:{device}"
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=dev)
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=dev)

    try:
        ms = do_bench(
            lambda: torch._int_mm(A, B),
            warmup=WARMUP,
            rep=REP,
        )
    except Exception as e:
        print(f"    INT8 _int_mm failed: {e}")
        return 0.0

    return _gemm_flops(M, N, K) / (ms * 1e-3) / 1e12


def _bench_mx_matrix_peak(
    label: str,
    bench_fn,
    device: int,
    shapes: List[Tuple[int, int, int]],
) -> float:
    print(f"\n  [{label}]  (Triton MX block-scaled GEMM; K divisible by {MX_BLOCK})")
    if not mx_available():
        print("    Not measured (Triton unavailable).")
        return 0.0
    best = 0.0
    measured = False
    for M, N, K in shapes:
        if K % MX_BLOCK != 0 or ("mxfp4" in label and K % 2 != 0):
            print(f"    ({M:>5},{N:>5},{K:>5}) → skipped (K alignment)")
            continue
        try:
            tflops = bench_fn(
                M, N, K, device, warmup=WARMUP, rep=REP, do_bench_fn=do_bench
            )
            print(f"    ({M:>5},{N:>5},{K:>5}) → {tflops:8.1f} TFLOPS")
            best = max(best, tflops)
            measured = True
        except Exception as e:
            print(f"    ({M:>5},{N:>5},{K:>5}) → skipped ({e})")
    if measured:
        print(f"    Best: {best:.1f}")
    else:
        print("    Not measured (no successful shapes).")
    return round(best, 1)


def bench_matrix_tflops(device: int = 0) -> Dict[str, float]:
    """Benchmark matrix (tensor core) TFLOPS across dtypes."""
    results = {}

    dtype_map = {
        "matrix_fp16": torch.float16,
        "matrix_bf16": torch.bfloat16,
        "matrix_fp32": torch.float32,
        "matrix_fp64": torch.float64,
    }

    for label, dtype in dtype_map.items():
        print(f"\n  [{label}]")
        shape_results = []
        for M, N, K in GEMM_SHAPES:
            tflops = bench_gemm(M, N, K, dtype, device)
            print(f"    ({M:>5},{N:>5},{K:>5}) → {tflops:8.1f} TFLOPS")
            shape_results.append(tflops)
        best = max(shape_results)
        median = sorted(shape_results)[len(shape_results) // 2]
        results[label] = round(best, 1)
        print(f"    Best: {best:.1f}  Median: {median:.1f}")

    # FP8
    print(f"\n  [matrix_fp8]")
    shape_results = []
    for M, N, K in GEMM_SHAPES:
        tflops = bench_gemm_fp8(M, N, K, device)
        print(f"    ({M:>5},{N:>5},{K:>5}) → {tflops:8.1f} TFLOPS")
        shape_results.append(tflops)
    best = max(shape_results)
    results["matrix_fp8"] = round(best, 1)
    print(f"    Best: {best:.1f}")

    triton_mxfp4 = _bench_mx_matrix_peak(
        "matrix_fp4 (triton dot_scaled)", bench_mxfp4_gemm, device, GEMM_SHAPES
    )
    ck_mxfp4 = _bench_mx_matrix_peak(
        "matrix_fp4 (aiter CK gemm_a4w4)", bench_mxfp4_ck_gemm, device, GEMM_SHAPES
    )
    results["matrix_fp4"] = round(max(triton_mxfp4, ck_mxfp4), 1)
    results["matrix_fp4_triton"] = triton_mxfp4
    results["matrix_fp4_ck"] = ck_mxfp4
    try:
        from .fp4fp6_helpers import (
            _MXFP6_KIND as _mxfp6_kind,
            _MXFP6_DTYPE as _mxfp6_dt,
        )
    except Exception:
        _mxfp6_kind, _mxfp6_dt = "", ""
    mxfp6_label = "matrix_fp6"
    if _mxfp6_kind == "f6f4_via_e2m1":
        mxfp6_label = (
            "matrix_fp6 (no native e3m2/e2m3 in Triton; running e2m1 which "
            "exercises the SAME v_mfma_scale_f32_*_f8f6f4 MFMA "
        )
    elif _mxfp6_kind == "native":
        mxfp6_label = f"matrix_fp6 (triton dot_scaled, native {_mxfp6_dt})"
    results["matrix_fp6"] = _bench_mx_matrix_peak(
        mxfp6_label, bench_mxfp6_gemm, device, GEMM_SHAPES
    )

    # INT8: torch._int_mm + aiter CK gemm_a8w8; take max.
    print("\n  [matrix_int8]  (torch._int_mm; TFLOPS = 2·M·N·K / time, AMD & CUDA)")
    torch_results = []
    for M, N, K in GEMM_SHAPES:
        tflops = bench_gemm_int8(M, N, K, device)
        print(f"    ({M:>5},{N:>5},{K:>5}) → {tflops:8.1f} TFLOPS")
        torch_results.append(tflops)
    torch_best = max(torch_results) if torch_results else 0.0
    print(f"    Best (torch._int_mm): {torch_best:.1f}")

    print("\n  [matrix_int8]  (aiter CK gemm_a8w8 → bf16 out)")
    ck_results = []
    for M, N, K in GEMM_SHAPES:
        tflops = bench_int8_ck_gemm(
            M, N, K, device, warmup=WARMUP, rep=REP, do_bench_fn=do_bench
        )
        print(f"    ({M:>5},{N:>5},{K:>5}) → {tflops:8.1f} TFLOPS")
        ck_results.append(tflops)
    ck_best = max(ck_results) if ck_results else 0.0
    print(f"    Best (aiter CK):       {ck_best:.1f}")

    best = max(torch_best, ck_best)
    results["matrix_int8"] = round(best, 1)
    winner = "aiter_ck" if ck_best > torch_best else "torch_int_mm"
    print(f"    Best overall:          {best:.1f}  ({winner})")

    return results


# ── Vector TFLOPS (Triton compute-bound FMA chain) ────────────────────

if triton is None or tl is None:

    def bench_vector_tflops(
        device: int = 0, *, repeat: int = 1024, mem_every: int = 0
    ) -> Dict[str, float]:
        print("\n── Vector TFLOPS (Triton FMA chain) ──")
        print("  Triton is not available; vector TFLOPS not measured.")
        return {
            "vector_fp16": 0.0,
            "vector_bf16": 0.0,
            "vector_fp32": 0.0,
            "vector_fp64": 0.0,
        }

else:

    @triton.jit
    def _fma_chain_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        REPEAT: tl.constexpr,
        ACC_FP32: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # Load once, then reuse in registers to raise arithmetic intensity.
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        y = tl.load(y_ptr + offs, mask=mask, other=0)
        acc = tl.load(z_ptr + offs, mask=mask, other=0)
        if ACC_FP32:
            x = x.to(tl.float32)
            y = y.to(tl.float32)
            acc = acc.to(tl.float32)

        # Dependency chain prevents the compiler from collapsing the loop into
        # a single multiply-by-constant.
        for _ in tl.static_range(REPEAT):
            acc = tl.math.fma(acc, x, y)  # acc = acc * x + y

        tl.store(out_ptr + offs, acc, mask=mask)

    @triton.jit
    def _fma_chain_with_stream_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        stream_ptr,
        out_ptr,
        n_elements,
        stream_pages,
        BLOCK_SIZE: tl.constexpr,
        REPEAT: tl.constexpr,
        MEM_EVERY: tl.constexpr,
        ACC_FP32: tl.constexpr,
    ):
        """
        Interleave a compute-bound FMA chain with streaming loads from a large
        buffer to estimate bandwidth while compute stays saturated.

        stream_ptr is conceptually shaped [stream_pages, n_elements]. We load a
        different "page" every MEM_EVERY iterations to reduce cache residency.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0)
        y = tl.load(y_ptr + offs, mask=mask, other=0)
        acc = tl.load(z_ptr + offs, mask=mask, other=0)
        if ACC_FP32:
            x = x.to(tl.float32)
            y = y.to(tl.float32)
            acc = acc.to(tl.float32)

        for i in tl.static_range(REPEAT):
            acc = tl.math.fma(acc, x, y)
            if MEM_EVERY > 0 and (i % MEM_EVERY) == 0:
                page = (i // MEM_EVERY) % stream_pages
                t = tl.load(stream_ptr + page * n_elements + offs, mask=mask, other=0)
                if ACC_FP32:
                    t = t.to(tl.float32)
                acc += t

        tl.store(out_ptr + offs, acc, mask=mask)

    def _triton_fma_chain_ms(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        out: torch.Tensor,
        *,
        repeat: int,
        block: int,
        acc_fp32: bool,
    ) -> float:
        n_elements = x.numel()
        grid = (triton.cdiv(n_elements, block),)
        ms = do_bench(
            lambda: _fma_chain_kernel[grid](
                x,
                y,
                z,
                out,
                n_elements,
                BLOCK_SIZE=block,
                REPEAT=repeat,
                ACC_FP32=(1 if acc_fp32 else 0),
                num_warps=8,
            ),
            warmup=WARMUP,
            rep=REP,
        )
        return ms

    def _triton_fma_chain_with_stream_ms(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        stream: torch.Tensor,
        out: torch.Tensor,
        *,
        repeat: int,
        mem_every: int,
        block: int,
        acc_fp32: bool,
    ) -> float:
        n_elements = x.numel()
        stream_pages = stream.numel() // n_elements
        grid = (triton.cdiv(n_elements, block),)
        ms = do_bench(
            lambda: _fma_chain_with_stream_kernel[grid](
                x,
                y,
                z,
                stream,
                out,
                n_elements,
                stream_pages,
                BLOCK_SIZE=block,
                REPEAT=repeat,
                MEM_EVERY=mem_every,
                ACC_FP32=(1 if acc_fp32 else 0),
                num_warps=8,
            ),
            warmup=WARMUP,
            rep=REP,
        )
        return ms

    def bench_triton_compute_bound(
        *,
        dtype: torch.dtype,
        n_elements: int,
        repeat: int,
        device: int = 0,
        block: int = 1024,
        mem_every: int = 0,
        acc_fp32: bool = False,
    ) -> Dict[str, float]:
        """
        Returns:
          - tflops: achieved TFLOPS for the FMA chain
          - stream_bw_gbps: achieved GB/s for the streaming loads (0 if disabled)
        """
        dev = f"cuda:{device}"

        x = torch.randn(n_elements, device=dev, dtype=dtype)
        y = torch.randn(n_elements, device=dev, dtype=dtype)
        z = torch.randn(n_elements, device=dev, dtype=dtype)
        out_dtype = torch.float32 if acc_fp32 else dtype
        out = torch.empty(n_elements, device=dev, dtype=out_dtype)

        if mem_every and mem_every > 0:
            stream_pages = max(1, (repeat + mem_every - 1) // mem_every)
            stream = torch.randn(stream_pages * n_elements, device=dev, dtype=dtype)
            ms = _triton_fma_chain_with_stream_ms(
                x,
                y,
                z,
                stream,
                out,
                repeat=repeat,
                mem_every=mem_every,
                block=block,
                acc_fp32=acc_fp32,
            )
            stream_bytes = stream_pages * n_elements * _bpe(dtype)
            stream_bw_gbps = stream_bytes / (ms * 1e-3) / 1e9
        else:
            ms = _triton_fma_chain_ms(
                x, y, z, out, repeat=repeat, block=block, acc_fp32=acc_fp32
            )
            stream_bw_gbps = 0.0

        flops = 2 * repeat * n_elements
        tflops = flops / (ms * 1e-3) / 1e12

        return {
            "tflops": float(tflops),
            "stream_bw_gbps": float(stream_bw_gbps),
            "ms": float(ms),
        }

    def bench_vector_tflops(
        device: int = 0, *, repeat: int = 1024, mem_every: int = 0
    ) -> Dict[str, float]:
        """
        Vector TFLOPS via a compute-bound Triton FMA dependency chain (not PyTorch
        elementwise). Keys match TraceLens: vector_fp16, vector_bf16, …
        Optional streaming loads (mem_every>0) print stream GB/s; they are not
        written to JSON (arch files only allow vector_* TFLOPS like `MI300X.json`).
        """
        results: Dict[str, float] = {}

        n_elements = 16 * 1024 * 1024  # 16M elements

        dtype_map = {
            "vector_fp16": torch.float16,
            "vector_bf16": torch.bfloat16,
            "vector_fp32": torch.float32,
            "vector_fp64": torch.float64,
        }

        print("\n── Vector TFLOPS (Triton FMA chain) ──")
        print(
            f"  n_elements={n_elements:,}  repeat={repeat}  mem_every={mem_every or 0}"
        )

        for label, dtype in dtype_map.items():
            acc_fp32 = (dtype == torch.float32) or (dtype == torch.bfloat16)
            r = bench_triton_compute_bound(
                dtype=dtype,
                n_elements=n_elements,
                repeat=repeat,
                device=device,
                mem_every=mem_every,
                acc_fp32=acc_fp32,
            )
            results[label] = round(r["tflops"], 1)
            print(
                f"  [{label}] {r['tflops']:7.1f} TFLOPS"
                + (
                    f" | stream {r['stream_bw_gbps']:7.1f} GB/s"
                    if mem_every and mem_every > 0
                    else ""
                )
                + f" | {r['ms']:.3f} ms"
            )

        return results


# ── HBM Bandwidth benchmark ──────────────────────────────────────────


def bench_hbm_bandwidth(
    device: int = 0,
    sizes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Measure HBM read and write bandwidth via tensor copy."""
    dev = f"cuda:{device}"
    nbytes_list = sizes if sizes is not None else BW_SIZES
    results: Dict[str, float] = {}

    print("\n  [HBM Read Bandwidth (copy src → dst)]")
    best_read = 0.0
    for nbytes in nbytes_list:
        n_elem = nbytes // 4  # float32
        src = torch.randn(n_elem, dtype=torch.float32, device=dev)
        dst = torch.empty_like(src)

        ms = do_bench(lambda: dst.copy_(src), warmup=WARMUP, rep=REP)
        # copy reads src + writes dst, so effective traffic is 2*nbytes.
        bw_gbps = (2 * nbytes) / (ms * 1e-3) / 1e9
        print(f"    {nbytes / 1e6:8.0f} MB  → {bw_gbps:8.1f} GB/s  ({ms:.3f} ms)")
        best_read = max(best_read, bw_gbps)

    results["read_bw_gbps"] = round(best_read, 1)
    print(f"    Best: {best_read:.1f} GB/s")

    print("\n  [HBM Write Bandwidth (fill)]")
    best_write = 0.0
    for nbytes in nbytes_list:
        n_elem = nbytes // 4
        dst = torch.empty(n_elem, dtype=torch.float32, device=dev)

        ms = do_bench(lambda: dst.fill_(1.0), warmup=WARMUP, rep=REP)
        bw_gbps = nbytes / (ms * 1e-3) / 1e9
        print(f"    {nbytes / 1e6:8.0f} MB  → {bw_gbps:8.1f} GB/s  ({ms:.3f} ms)")
        best_write = max(best_write, bw_gbps)

    results["write_bw_gbps"] = round(best_write, 1)
    print(f"    Best: {best_write:.1f} GB/s")

    return results


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "hip out of memory" in msg


def _sweep_gemm_at_shape(
    M: int, N: int, K: int, device: int
) -> List[Dict[str, object]]:
    """Run all matrix dtypes at one (M, N, K); return per-metric rows."""
    rows: List[Dict[str, object]] = []
    shape = {"M": M, "N": N, "K": K}

    def row(
        metric: str, tflops: float, error: Optional[str] = None
    ) -> Dict[str, object]:
        return {**shape, "metric": metric, "tflops": round(tflops, 1), "error": error}

    dtype_map = {
        "matrix_fp16": torch.float16,
        "matrix_bf16": torch.bfloat16,
        "matrix_fp32": torch.float32,
        "matrix_fp64": torch.float64,
    }
    for label, dtype in dtype_map.items():
        try:
            torch.cuda.empty_cache()
            tflops = bench_gemm(M, N, K, dtype, device)
            rows.append(row(label, tflops))
        except RuntimeError as e:
            rows.append(row(label, 0.0, "oom" if _is_oom(e) else str(e)))

    try:
        torch.cuda.empty_cache()
        tflops = bench_gemm_fp8(M, N, K, device)
        rows.append(row("matrix_fp8", tflops))
    except RuntimeError as e:
        rows.append(row("matrix_fp8", 0.0, "oom" if _is_oom(e) else str(e)))

    try:
        torch.cuda.empty_cache()
        tflops_torch = bench_gemm_int8(M, N, K, device)
        tflops_ck = bench_int8_ck_gemm(
            M, N, K, device, warmup=WARMUP, rep=REP, do_bench_fn=do_bench
        )
        rows.append(row("matrix_int8", max(tflops_torch, tflops_ck)))
    except RuntimeError as e:
        rows.append(row("matrix_int8", 0.0, "oom" if _is_oom(e) else str(e)))

    if mx_available() and K % MX_BLOCK == 0 and K % 2 == 0:
        for label, fn in (
            ("matrix_fp4", bench_mxfp4_gemm),
            ("matrix_fp6", bench_mxfp6_gemm),
        ):
            try:
                torch.cuda.empty_cache()
                tflops = fn(
                    M, N, K, device, warmup=WARMUP, rep=REP, do_bench_fn=do_bench
                )
                rows.append(row(label, tflops))
            except Exception as e:
                rows.append(row(label, 0.0, "oom" if _is_oom(e) else str(e)))

    return rows


def _best_per_metric(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    best: Dict[str, Dict[str, object]] = {}
    for r in rows:
        if r.get("error"):
            continue
        metric = str(r["metric"])
        tflops = float(r["tflops"])
        if metric not in best or tflops > float(best[metric]["tflops"]):
            best[metric] = {
                "M": r["M"],
                "N": r["N"],
                "K": r["K"],
                "tflops": tflops,
            }
    return best


def _sweep_shapes_on_list(
    shapes: List[Tuple[int, int, int]],
    device: int,
    label: str,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    all_rows: List[Dict[str, object]] = []
    print(f"\n── GEMM sweep: {label} ({len(shapes)} shapes) ──")
    for M, N, K in shapes:
        print(f"\n  Shape ({M:>5}, {N:>5}, {K:>5})")
        try:
            rows = _sweep_gemm_at_shape(M, N, K, device)
            all_rows.extend(rows)
            for r in rows:
                err = r.get("error")
                if err:
                    print(f"    [{r['metric']}] skipped ({err})")
                else:
                    print(f"    [{r['metric']}] {r['tflops']:8.1f} TFLOPS")
        except RuntimeError as e:
            err = "oom" if _is_oom(e) else str(e)
            print(f"    all metrics skipped ({err})")
            for metric in (
                "matrix_fp16",
                "matrix_bf16",
                "matrix_fp32",
                "matrix_fp64",
                "matrix_fp8",
                "matrix_int8",
            ):
                all_rows.append(
                    {
                        "M": M,
                        "N": N,
                        "K": K,
                        "metric": metric,
                        "tflops": 0.0,
                        "error": err,
                    }
                )
    return all_rows, _best_per_metric(all_rows)


def bench_hbm_bandwidth_sweep(
    device: int = 0,
    sizes: Optional[List[int]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Per-size HBM read/write; returns rows and bests."""
    nbytes_list = sizes if sizes is not None else BW_SIZES_SWEEP_LARGE
    dev = f"cuda:{device}"
    rows: List[Dict[str, object]] = []
    best_read = 0.0
    best_write = 0.0
    best_read_n = 0
    best_write_n = 0

    print(f"\n── HBM sweep: large transfers ({len(nbytes_list)} sizes) ──")
    for nbytes in nbytes_list:
        n_elem = nbytes // 4
        read_gbps = 0.0
        write_gbps = 0.0
        err: Optional[str] = None
        try:
            torch.cuda.empty_cache()
            src = torch.randn(n_elem, dtype=torch.float32, device=dev)
            dst = torch.empty_like(src)
            ms = do_bench(lambda: dst.copy_(src), warmup=WARMUP, rep=REP)
            read_gbps = (2 * nbytes) / (ms * 1e-3) / 1e9
        except RuntimeError as e:
            err = "oom" if _is_oom(e) else str(e)

        try:
            torch.cuda.empty_cache()
            dst = torch.empty(n_elem, dtype=torch.float32, device=dev)
            ms = do_bench(lambda: dst.fill_(1.0), warmup=WARMUP, rep=REP)
            write_gbps = nbytes / (ms * 1e-3) / 1e9
        except RuntimeError as e:
            err = err or ("oom" if _is_oom(e) else str(e))

        rows.append(
            {
                "nbytes": nbytes,
                "nbytes_gb": round(nbytes / 1024**3, 2),
                "read_gbps": round(read_gbps, 1),
                "write_gbps": round(write_gbps, 1),
                "error": err,
            }
        )
        if read_gbps > best_read:
            best_read, best_read_n = read_gbps, nbytes
        if write_gbps > best_write:
            best_write, best_write_n = write_gbps, nbytes
        tag = f" ({err})" if err else ""
        print(
            f"  {nbytes / 1024**3:6.1f} GB  read {read_gbps:8.1f} GB/s  "
            f"write {write_gbps:8.1f} GB/s{tag}"
        )

    bests = {
        "read_bw_gbps": round(best_read, 1),
        "write_bw_gbps": round(best_write, 1),
        "best_read_nbytes": best_read_n,
        "best_write_nbytes": best_write_n,
    }
    print(f"  Best read:  {best_read:.1f} GB/s @ {best_read_n / 1024**3:.1f} GB")
    print(f"  Best write: {best_write:.1f} GB/s @ {best_write_n / 1024**3:.1f} GB")
    return rows, bests


def _pick_best_with_source(
    *labeled_bests: Tuple[str, Dict[str, Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    """Per metric, keep the highest TFLOPS across labeled best dicts."""
    out: Dict[str, Dict[str, object]] = {}
    for source, bests in labeled_bests:
        for metric, info in bests.items():
            tflops = float(info.get("tflops", 0.0))
            if metric not in out or tflops > float(out[metric]["tflops"]):
                out[metric] = {
                    "M": info.get("M"),
                    "N": info.get("N"),
                    "K": info.get("K"),
                    "tflops": tflops,
                    "source": source,
                }
    return out


def run_shape_sweep(
    device: int,
    output_path: Path,
    *,
    include_large: bool = True,
    include_tile304: bool = True,
    include_hbm: bool = True,
    include_production: bool = True,
) -> Dict[str, object]:
    """
    Sweep candidate GEMM shapes and optional HBM sizes; compare to production ``GEMM_SHAPES``.
    """
    global _FP8_DTYPE_CACHE
    _FP8_DTYPE_CACHE = None

    prod_rows: List[Dict[str, object]] = []
    prod_best: Dict[str, Dict[str, object]] = {}
    if include_production:
        prod_rows, prod_best = _sweep_shapes_on_list(
            GEMM_SHAPES, device, "production GEMM_SHAPES"
        )

    large_rows: List[Dict[str, object]] = []
    large_best: Dict[str, Dict[str, object]] = {}
    if include_large:
        large_rows, large_best = _sweep_shapes_on_list(
            GEMM_SHAPES_SWEEP_LARGE, device, "large GEMM candidates"
        )

    tile304_rows: List[Dict[str, object]] = []
    tile304_best: Dict[str, Dict[str, object]] = {}
    if include_tile304:
        tile304_rows, tile304_best = _sweep_shapes_on_list(
            GEMM_SHAPES_SWEEP_TILE304, device, "tile-304 sweet shapes"
        )

    hbm_rows: List[Dict[str, object]] = []
    hbm_bests: Dict[str, float] = {}
    if include_hbm:
        hbm_rows, hbm_bests = bench_hbm_bandwidth_sweep(device, BW_SIZES_SWEEP_LARGE)

    candidate_best = _pick_best_with_source(
        ("large", large_best),
        ("tile304", tile304_best),
    )

    print("\n── Sweep vs production peaks ──")
    comparison: Dict[str, Dict[str, object]] = {}
    all_metrics = sorted(set(prod_best) | set(candidate_best))
    for metric in all_metrics:
        p = prod_best.get(metric, {})
        c = candidate_best.get(metric, {})
        pt = float(p.get("tflops", 0.0))
        ct = float(c.get("tflops", 0.0))
        delta = ct - pt
        winner = str(c.get("source", "production")) if ct > pt else "production"
        lg = large_best.get(metric, {})
        t3 = tile304_best.get(metric, {})
        comparison[metric] = {
            "production_tflops": pt,
            "production_shape": [p.get("M"), p.get("N"), p.get("K")],
            "candidate_tflops": ct,
            "candidate_shape": [c.get("M"), c.get("N"), c.get("K")],
            "candidate_source": c.get("source"),
            "large_tflops": float(lg.get("tflops", 0.0)),
            "large_shape": [lg.get("M"), lg.get("N"), lg.get("K")],
            "tile304_tflops": float(t3.get("tflops", 0.0)),
            "tile304_shape": [t3.get("M"), t3.get("N"), t3.get("K")],
            "delta_tflops": round(delta, 1),
            "winner": winner,
        }
        print(
            f"  {metric:<14} prod {pt:8.1f}  best {ct:8.1f} ({winner})  "
            f"Δ {delta:+6.1f}"
        )

    payload: Dict[str, object] = {
        "sweep_type": "gemm_and_hbm",
        "warmup": WARMUP,
        "rep": REP,
        "production_gemm_shapes": [list(s) for s in GEMM_SHAPES],
        "large_gemm_shapes": [list(s) for s in GEMM_SHAPES_SWEEP_LARGE],
        "tile304_gemm_shapes": [list(s) for s in GEMM_SHAPES_SWEEP_TILE304],
        "large_bw_sizes_gb": [n / 1024**3 for n in BW_SIZES_SWEEP_LARGE],
        "production_gemm_rows": prod_rows,
        "production_gemm_best": prod_best,
        "large_gemm_rows": large_rows,
        "large_gemm_best": large_best,
        "tile304_gemm_rows": tile304_rows,
        "tile304_gemm_best": tile304_best,
        "candidate_gemm_best": candidate_best,
        "comparison": comparison,
        "hbm_rows": hbm_rows,
        "hbm_best": hbm_bests,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("M,N,K,metric,tflops,error,sweep_set\n")
        for r in prod_rows:
            f.write(
                f"{r['M']},{r['N']},{r['K']},{r['metric']},{r['tflops']},"
                f"{r.get('error') or ''},production\n"
            )
        for r in large_rows:
            f.write(
                f"{r['M']},{r['N']},{r['K']},{r['metric']},{r['tflops']},"
                f"{r.get('error') or ''},large\n"
            )
        for r in tile304_rows:
            f.write(
                f"{r['M']},{r['N']},{r['K']},{r['metric']},{r['tflops']},"
                f"{r.get('error') or ''},tile304\n"
            )
    print(f"\nSweep written to {output_path}")
    print(f"CSV written to {csv_path}")
    return payload


# ── Main ──────────────────────────────────────────────────────────────


def main():
    global WARMUP, REP

    parser = argparse.ArgumentParser(description="GPU Microbenchmarking Suite")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_microbench_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="Skip vector TFLOPS benchmarks (Triton FMA chain)",
    )
    parser.add_argument(
        "--triton-repeat",
        type=int,
        default=1024,
        help="Repeat count for Triton vector FMA chain (higher = more compute-bound)",
    )
    parser.add_argument(
        "--triton-mem-every",
        type=int,
        default=0,
        help="If >0, do one streaming load every N iterations to estimate bandwidth at peak compute",
    )
    parser.add_argument(
        "--skip-bandwidth", action="store_true", help="Skip HBM bandwidth benchmarks"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override do_bench warmup iterations (default: 30)",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=None,
        help="Override do_bench timing repetitions (default: 200)",
    )
    parser.add_argument(
        "--shape-sweep",
        action="store_true",
        help="Sweep large GEMM shapes + multi-GB HBM sizes (see GEMM_SHAPES_SWEEP_LARGE)",
    )
    parser.add_argument(
        "--sweep-output",
        type=str,
        default=None,
        help="JSON path for --shape-sweep (default: results/shape_sweep_<timestamp>.json)",
    )
    parser.add_argument(
        "--shape-sweep-tile304-only",
        action="store_true",
        help="With --shape-sweep: only production + tile-304 shapes (skip large/HBM)",
    )
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="Skip the pre-flight idle check and run even if the GPU is busy",
    )
    parser.add_argument(
        "--idle-util-threshold",
        type=int,
        default=5,
        help="Max GPU utilization %% considered idle for the pre-flight check (default: 5)",
    )
    args = parser.parse_args()

    if args.warmup is not None:
        WARMUP = args.warmup
    if args.rep is not None:
        REP = args.rep

    device = args.device
    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    mem_gb = round(total_mem / 1024**3, 1)

    print("=" * 60)
    print(f"GPU Microbenchmarking Suite")
    print(f"=" * 60)
    print(f"Device:    {gpu_name} (index {device})")
    print(f"Memory:    {mem_gb} GB")
    print(f"PyTorch:   {torch.__version__}")
    print(f"ROCm/HIP:  {torch.version.hip}")
    print(f"Warmup:    {WARMUP}  |  Repetitions: {REP}")
    print("=" * 60)

    idle, msg = check_gpu_idle(device, util_threshold=args.idle_util_threshold)
    print(f"Idle check: {msg}")
    if not idle:
        if args.allow_busy:
            print("Warning: GPU appears busy but --allow-busy is set; continuing.")
        else:
            print(
                "ERROR: GPU is not idle. Stop other workloads on this device or "
                "rerun with --allow-busy to override.",
            )
            import sys

            sys.exit(2)
    print("=" * 60)

    if args.shape_sweep:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sweep_out = (
            Path(args.sweep_output)
            if args.sweep_output
            else Path(f"results/shape_sweep_{ts}.json")
        )
        if not sweep_out.is_absolute():
            sweep_out = Path(__file__).resolve().parent / sweep_out
        run_shape_sweep(
            device,
            sweep_out,
            include_large=not args.shape_sweep_tile304_only,
            include_tile304=True,
            include_hbm=not args.shape_sweep_tile304_only,
            include_production=True,
        )
        return

    # Matrix TFLOPS
    print("\n── Matrix TFLOPS (MFMA Units & Tensor Cores) ──")
    matrix_results = bench_matrix_tflops(device)

    # Vector TFLOPS (Triton)
    if not args.skip_vector:
        vector_results = bench_vector_tflops(
            device,
            repeat=args.triton_repeat,
            mem_every=args.triton_mem_every,
        )
    else:
        vector_results = {}

    # HBM Bandwidth
    if not args.skip_bandwidth:
        print("\n── HBM Bandwidth ──")
        bw_results = bench_hbm_bandwidth(device)
    else:
        bw_results = {}

    read_bw = float(bw_results.get("read_bw_gbps", 0.0))
    write_bw = float(bw_results.get("write_bw_gbps", 0.0))
    mem_bw = max(read_bw, write_bw)
    arch_json = _build_measured_arch_json(
        gpu_name, mem_gb, mem_bw, matrix_results, vector_results
    )

    # JSON: same shape as `results/MI300X.json`.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(arch_json, f, indent=4)
    print(f"\nResults written to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'Measured':>12} {'Unit':>8}")
    print("-" * 42)
    for k, v in matrix_results.items():
        print(f"{k:<20} {v:>12.1f} {'TFLOPS':>8}")
    for k, v in vector_results.items():
        print(f"{k:<20} {v:>12.1f} {'TFLOPS':>8}")
    for k, v in bw_results.items():
        print(f"{k:<20} {v:>12.1f} {'GB/s':>8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
