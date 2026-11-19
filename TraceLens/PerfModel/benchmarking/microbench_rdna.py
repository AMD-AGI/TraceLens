###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RDNA (gfx11, gfx12) measurement protocol of the GPU microbenchmark suite.

:mod:`microbench` runs :func:`run_rdna_protocol` on RDNA targets instead of its
own sections, so that keys are comparable across boards and runs:

- Timing: about 3 s of back-to-back calls before each section, then a 1 s
  ``do_bench`` window per config reported as the median, with the sysfs shader,
  memory and fabric clocks, power and power cap of that window.
- Matrix: the best library GEMM over hipBLASLt and rocBLAS in the NN and NT
  layouts, over the suite's shapes plus two deep-K shapes for the WMMA formats.
  A Triton WMMA loop on register tiles is recorded next to the keys with its
  clock; it is not a key.
- Vector: the best of the suite's FMA chain and an [ILP, BLOCK] accumulator
  tile updated as ``fma(x, y, acc)``, which dual-issues as VOPD.
- Bandwidth: the read-only plateau of a Triton read kernel and of
  ``torch.sum``, the higher of the two, each call after a read-only cache
  flush. Copy and fill are recorded as diagnostics.
- Side measurements, recorded but not published: each winner timed for 100 ms
  after 5 s idle (``burst``), and the 16-bit GEMM winners run back to back for
  30 s (``back_to_back``).
- Keys: :func:`rdna_published_keys`. A zero or failed key the target has a
  path for raises :class:`MeasurementError`, and no arch JSON is written.
"""

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
from triton.testing import do_bench

from .microbench_utils import (
    BW_PLATEAU_TOL,
    RDNA_BACK_TO_BACK_CHUNK_S,
    RDNA_BACK_TO_BACK_FIRST_CHUNK_S,
    RDNA_BACK_TO_BACK_S,
    RDNA_BACK_TO_BACK_SETTLE_S,
    RDNA_BURST_IDLE_S,
    RDNA_BURST_WINDOW_MS,
    RDNA_REWARM_AFTER_S,
    RDNA_REWARM_S,
    RDNA_WARM_S,
    RDNA_WINDOW_MS,
    TELEMETRY_PERIOD_S,
    MeasurementError,
    SysfsTelemetry,
    bandwidth_plateau,
    check_rdna_gpu_idle,
    pci_bdf,
    platform_record,
    power_held_w,
    rdna_measured_keys,
    rdna_published_keys,
    vopd_share,
)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# Errors a config can fail with (HIP or library errors, out of memory, a kernel
# that does not compile or fit); the config is skipped.
_CONFIG_ERRORS: Tuple[type, ...] = (RuntimeError,)
if triton is not None:
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    _CONFIG_ERRORS += (CompilationError, OutOfResources)

# The do_bench warm-up inside each window; the section warm-up comes before it.
DO_BENCH_WARMUP_MS = 30
# GEMM libraries, as (name, ``torch.backends.cuda.preferred_blas_library``
# value), and layouts every floating-point matrix key takes the best of. ``nt``
# is the layout ``nn.Linear`` passes; on gfx1100 it is hipBLASLt's faster one,
# rocBLAS's slower.
RDNA_BLAS_LIBRARIES: Tuple[Tuple[str, str], ...] = (
    ("hipblaslt", "cublaslt"),
    ("rocblas", "cublas"),
)
RDNA_GEMM_LAYOUTS: Tuple[str, ...] = ("nn", "nt")
# Shapes added to the suite's for the WMMA formats: the fastest found on gfx1151
# (2560x1920x32896, hipBLASLt) and on gfx1201 (4096x4096x16512, NT). The suite's
# are wide and shallow, sized for MI300X's 304 CUs. K is a multiple of 128 but
# not a power of two, which in NT aliases in cache.
RDNA_DEEP_K_SHAPES: List[Tuple[int, int, int]] = [
    (2560, 1920, 32896),
    (4096, 4096, 16512),
]
# FP64 GEMMs run on the vector ALUs at about 1/32 of the FP32 rate, so a suite
# shape takes seconds per call; more shapes add run time, not signal.
RDNA_FP64_SHAPES: List[Tuple[int, int, int]] = [(4096, 4096, 4096)]
# Vector FMA kernels, best of: the suite's chain ``fma(acc, x, y)`` at these
# (BLOCK, num_warps), which include the suite's own launch (its
# ``vector_fma_block`` with 8 warps), and a [ILP, BLOCK] tile of accumulators
# updated as ``fma(x, y, acc)`` with x and y broadcast over ILP. Only the
# tile's form compiles to ``v_fmac_f32``, which VOPD pairs (wave32 dual
# issue); at 2 elements per lane it pairs every FMA on gfx1100, gfx1151 and
# gfx1201. The tile runs the chain's FLOPs over 1/ILP of its elements, each
# with ILP chains of REPEAT FMAs in a loop unrolled by VECTOR_TILE_UNROLL: at
# the chain's element count its loads and stores take half of gfx1201's DRAM
# bandwidth.
VECTOR_CONFIGS_RDNA: List[Tuple[int, int]] = [
    (256, 4),
    (256, 8),
    (512, 4),
    (512, 8),
    (1024, 8),
    (2048, 8),
]
VECTOR_TILE_ILP = 8
VECTOR_TILE_UNROLL = 16
VECTOR_TILE_CONFIGS_RDNA: List[Tuple[int, int]] = [
    (256, 4),
    (512, 8),
    (256, 8),
    (512, 4),
]
VECTOR_N_ELEMENTS = 16 * 1024 * 1024
# WMMA loop on register tiles (no memory traffic), recorded next to the matrix
# keys as (BM, BN, BK, num_warps); programs and work per launch.
WMMA_LOOP_CONFIGS: List[Tuple[int, int, int, int]] = [(32, 32, 32, 2), (64, 64, 32, 2)]
WMMA_LOOP_PROGRAMS = 4096
WMMA_LOOP_FLOP_PER_LAUNCH = 2e12
WMMA_LOOP_UNROLL = 4
# Read-only bandwidth: buffer sizes (capped at half the free memory; each at
# least 4x the largest RDNA last-level cache, 96 MB), the read-only sweep that
# evicts the caches before each call without leaving dirty lines, the read
# kernel's (BLOCK, grid) candidates, and the calls per size.
BW_SIZES_RDNA: List[int] = [2**k * 512 * 1024 * 1024 for k in range(5)]  # 512 MB-8 GB
BW_FLUSH_BYTES = 512 * 1024 * 1024
BW_READ_CONFIGS: List[Tuple[int, int]] = [
    (2048, 1024),
    (4096, 1024),
    (4096, 2048),
    (8192, 1024),
]
BW_MIN_CALLS = 10
BW_MAX_CALLS = 2000
BW_TUNE_WINDOW_MS = 50
# Copy and fill diagnostics: the suite's sizes, 64 MB-2 GB.
BW_DIAG_SIZES: List[int] = [2**k * 64 * 1024 * 1024 for k in range(6)]
# Winners run back to back after the sections.
BACK_TO_BACK_KEYS: Tuple[str, ...] = ("matrix_fp16", "matrix_bf16")


@dataclass(frozen=True)
class SuiteParts:
    """
    What the protocol reuses from :mod:`microbench`, passed in because this
    module may not import it (import-linter forbids the cycle): the suite's
    GEMM shapes, its FMA chain kernel (the suite's launch is one of the
    candidates), its FP8 dtype probe for ``torch._scaled_mm``, and the chain's
    compile-time arguments besides ``BLOCK_SIZE``, ``REPEAT`` and ``ACC_FP32``,
    as the suite passes them.
    """

    gemm_shapes: Sequence[Tuple[int, int, int]]
    chain_kernel: Any
    resolve_fp8_dtype: Callable[[int, int, int, int], Optional[torch.dtype]]
    chain_args: Mapping[str, Any] = field(default_factory=dict)


if triton is not None and tl is not None:

    @triton.jit
    def _fma_tile_kernel(
        x_ptr,
        y_ptr,
        z_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        ILP: tl.constexpr,
        REPEAT: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        y = tl.load(y_ptr + offs, mask=mask, other=0)
        z = tl.load(z_ptr + offs, mask=mask, other=0)
        xb = tl.broadcast_to(x[None, :], (ILP, BLOCK_SIZE))
        yb = tl.broadcast_to(y[None, :], (ILP, BLOCK_SIZE))
        # ILP independent chains per element, REPEAT FMAs per chain.
        acc = z[None, :] + tl.arange(0, ILP)[:, None].to(z.dtype)
        for _ in range(REPEAT // UNROLL):
            for _ in tl.static_range(UNROLL):
                acc = tl.math.fma(xb, yb, acc)
        tl.store(out_ptr + offs, tl.sum(acc, axis=0), mask=mask)

    @triton.jit
    def _wmma_loop_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        ITERS,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        UNROLL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * BK + rk[None, :])
        b = tl.load(b_ptr + rk[:, None] * BN + rn[None, :])
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(ITERS // UNROLL):
            for _ in tl.static_range(UNROLL):
                acc = tl.dot(a, b, acc)
        tl.store(c_ptr + pid * BM * BN + rm[:, None] * BN + rn[None, :], acc)

    @triton.jit
    def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        nprog = tl.num_programs(0)
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for start in range(pid * BLOCK, n, nprog * BLOCK):
            offs = start + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0)
        tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), acc)


@dataclass
class _Candidate:
    """One config a key takes the best of. *factory* allocates the inputs and
    returns the call to time; *work* is FLOPs (or bytes) per call; *blas* is
    the ``preferred_blas_library`` it runs under."""

    config: Dict[str, Any]
    factory: Callable[[], Callable[[], Any]]
    work: float
    blas: Optional[str] = None


@dataclass
class _Winner:
    """A section's winning config and rate, for the side measurements."""

    section: str
    key: str
    candidate: _Candidate
    rate_key: str
    rate: float
    flush: Optional[Callable[[], Any]] = None


_RATE_UNITS = {"tflops": ("TFLOPS", 1e12), "gbps": ("GB/s", 1e9)}


@contextmanager
def _preferred_blas(backend: Optional[str]) -> Iterator[None]:
    if backend is None:
        yield
        return
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(backend)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def _config_label(config: Dict[str, Any]) -> str:
    parts = []
    for name, value in config.items():
        if isinstance(value, (list, tuple)):
            parts.append("x".join(str(v) for v in value))
        elif isinstance(value, str):
            parts.append(value)
        else:
            parts.append(f"{name}={value}")
    return " ".join(parts)


def _error_text(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0][:160]


def _gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def _time_clean(
    fn: Callable[[], Any], flush: Callable[[], Any], window_ms: float
) -> float:
    """Median ms per call of *fn* with a read-only *flush* before each call, over
    about *window_ms* of calls."""
    flush()
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    flush()
    fn()
    torch.cuda.synchronize()
    per_call_s = max(time.perf_counter() - start, 1e-6)
    calls = min(BW_MAX_CALLS, max(BW_MIN_CALLS, int(window_ms * 1e-3 / per_call_s)))
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(calls)
    ]
    for begin, end in events:
        flush()
        begin.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(begin.elapsed_time(end) for begin, end in events)


class _RdnaProtocol:
    """
    Timing on RDNA targets: about *warm_s* of back-to-back calls before each
    section, then a ``do_bench`` window of *window_ms* reporting the median ms
    per call, with the sysfs clocks of that window.
    """

    def __init__(
        self, telemetry: SysfsTelemetry, *, warm_s: float, window_ms: int
    ) -> None:
        self.telemetry = telemetry
        self.warm_s = warm_s
        self.window_ms = window_ms

    def warm(self, fn: Callable[[], Any], seconds: Optional[float] = None) -> None:
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        batch = max(1, int(0.05 / max(time.perf_counter() - start, 1e-6)))
        while time.perf_counter() - start < (seconds or self.warm_s):
            for _ in range(batch):
                fn()
            torch.cuda.synchronize()

    def clocks(self, t0: float, t1: float) -> Dict[str, Any]:
        return self.telemetry.summary(t0, t1) if self.telemetry.available else {}

    def time(
        self, fn: Callable[[], Any], window_ms: Optional[int] = None
    ) -> Tuple[float, Dict[str, Any]]:
        t0 = time.time()
        ms = do_bench(
            fn,
            warmup=DO_BENCH_WARMUP_MS,
            rep=window_ms or self.window_ms,
            return_mode="median",
        )
        return ms, self.clocks(t0, time.time())

    def best_of(
        self,
        candidates: List[_Candidate],
        *,
        rate_key: str = "tflops",
        timer: Optional[Callable[[Callable[[], Any]], Tuple[float, Dict]]] = None,
        warm: bool = True,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Times every candidate, after warming the card with the first that runs
        (unless *warm* is off). Each candidate runs once before its window, so a
        Triton compile is not timed; when its setup took over
        ``RDNA_REWARM_AFTER_S`` it is warmed for ``RDNA_REWARM_S`` first.
        Returns the best as ``{rate, config, clock, candidate}`` (``None`` if
        no candidate ran) and a row per candidate.
        """
        unit, scale = _RATE_UNITS[rate_key]
        timer = timer or self.time
        rows: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        for cand in candidates:
            label = _config_label(cand.config)
            fn = None
            try:
                with _preferred_blas(cand.blas):
                    setup_start = time.perf_counter()
                    fn = cand.factory()
                    fn()
                    torch.cuda.synchronize()
                    if warm:
                        self.warm(fn)
                        warm = False
                    elif time.perf_counter() - setup_start > RDNA_REWARM_AFTER_S:
                        self.warm(fn, RDNA_REWARM_S)
                    ms, clock = timer(fn)
            except _CONFIG_ERRORS as e:
                print(f"    {label:<44} failed: {_error_text(e)}")
                rows.append({**cand.config, "error": _error_text(e)})
                continue
            finally:
                del fn
            rate = cand.work / (ms * 1e-3) / scale
            rows.append(
                {**cand.config, rate_key: round(rate, 1), "ms": round(ms, 4), **clock}
            )
            print(f"    {label:<44} {rate:8.1f} {unit}  ({ms:.3f} ms)")
            if best is None or rate > best["rate"]:
                best = {"rate": rate, "config": cand.config, "clock": clock}
                best["candidate"] = cand
        return best, rows


def _clock_of(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("samples", "sclk_mhz", "mclk_mhz", "fclk_mhz", "power_w", "power_cap_w")
    return {k: row[k] for k in keys if k in row}


def _section(
    best: Optional[Dict[str, Any]], rows: List[Dict[str, Any]], rate_key: str
) -> Dict[str, Any]:
    if best is None:
        return {rate_key: 0.0, "rows": rows}
    return {
        rate_key: round(best["rate"], 1),
        "config": best["config"],
        "clock": best["clock"],
        "rows": rows,
    }


# ── Matrix ───────────────────────────────────────────────────────────


def _gemm_call(M: int, N: int, K: int, dtype: torch.dtype, dev: str, layout: str):
    A = torch.randn(M, K, dtype=dtype, device=dev)
    if layout == "nt":
        B = torch.randn(N, K, dtype=dtype, device=dev)
        return lambda: torch.matmul(A, B.t())
    B = torch.randn(K, N, dtype=dtype, device=dev)
    return lambda: torch.matmul(A, B)


def _int8_call(M: int, N: int, K: int, dev: str, layout: str):
    A = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=dev)
    if layout == "nt":
        B = torch.randint(-128, 127, (N, K), dtype=torch.int8, device=dev)
        return lambda: torch._int_mm(A, B.t())
    B = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=dev)
    return lambda: torch._int_mm(A, B)


def _fp8_call(M: int, N: int, K: int, dev: str, fp8_dtype: torch.dtype):
    a = torch.randn(M, K, device=dev).to(fp8_dtype)
    b = torch.randn(N, K, device=dev).to(fp8_dtype)
    scale = torch.ones(1, dtype=torch.float32, device=dev)
    return lambda: torch._scaled_mm(
        a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16
    )


def _library_gemms(
    dtype: torch.dtype, shapes: Sequence[Tuple[int, int, int]], dev: str
) -> List[_Candidate]:
    return [
        _Candidate(
            {"library": lib, "layout": layout, "shape": [M, N, K]},
            partial(_gemm_call, M, N, K, dtype, dev, layout),
            _gemm_flops(M, N, K),
            blas=backend,
        )
        for lib, backend in RDNA_BLAS_LIBRARIES
        for layout in RDNA_GEMM_LAYOUTS
        for M, N, K in shapes
    ]


def _matrix_plans(
    device: int, gfx: str, suite: SuiteParts
) -> Dict[str, List[_Candidate]]:
    """The configs each matrix key an RDNA target has a path for takes the best
    of: FP16, BF16, FP32 and FP64 through both GEMM libraries in both layouts,
    INT8 through ``torch._int_mm`` in both layouts, and FP8 through
    ``torch._scaled_mm`` on gfx12."""
    dev = f"cuda:{device}"
    shapes = list(suite.gemm_shapes)
    wmma_shapes = shapes + [s for s in RDNA_DEEP_K_SHAPES if s not in shapes]
    plans = {
        "matrix_fp16": _library_gemms(torch.float16, wmma_shapes, dev),
        "matrix_bf16": _library_gemms(torch.bfloat16, wmma_shapes, dev),
        "matrix_fp32": _library_gemms(torch.float32, shapes, dev),
        "matrix_fp64": _library_gemms(torch.float64, RDNA_FP64_SHAPES, dev),
        "matrix_int8": [
            _Candidate(
                {"library": "torch._int_mm", "layout": layout, "shape": [M, N, K]},
                partial(_int8_call, M, N, K, dev, layout),
                _gemm_flops(M, N, K),
            )
            for layout in RDNA_GEMM_LAYOUTS
            for M, N, K in wmma_shapes
        ],
    }
    if "matrix_fp8" in rdna_measured_keys(gfx):
        fp8_dtype = suite.resolve_fp8_dtype(device, *shapes[0])
        plans["matrix_fp8"] = [
            _Candidate(
                {"library": "torch._scaled_mm", "layout": "nt", "shape": [M, N, K]},
                partial(_fp8_call, M, N, K, dev, fp8_dtype),
                _gemm_flops(M, N, K),
            )
            for M, N, K in wmma_shapes
            if fp8_dtype is not None
        ]
    return plans


def bench_matrix_rdna(
    device: int, gfx: str, proto: _RdnaProtocol, suite: SuiteParts
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], List[_Winner]]:
    """
    The matrix keys an RDNA target has a path for (FP16, BF16, FP32, FP64,
    INT8, and FP8 on gfx12), each the best over :func:`_matrix_plans`. The
    other formats are derived by :func:`rdna_published_keys`.
    """
    results: Dict[str, float] = {}
    sections: Dict[str, Dict[str, Any]] = {}
    winners: List[_Winner] = []
    for key, candidates in _matrix_plans(device, gfx, suite).items():
        print(f"\n  [{key}]  best of {len(candidates)} configs")
        best, rows = proto.best_of(candidates)
        results[key] = round(best["rate"], 1) if best else 0.0
        sections[key] = _section(best, rows, "tflops")
        if best:
            winners.append(
                _Winner("matrix", key, best["candidate"], "tflops", best["rate"])
            )
    return results, sections, winners


def _wmma_loop_call(
    bm: int, bn: int, bk: int, warps: int, iters: int, dtype: torch.dtype, dev: str
):
    a = (torch.randn(bm, bk, device=dev) / 8).to(dtype)
    b = (torch.randn(bk, bn, device=dev) / 8).to(dtype)
    c = torch.empty(WMMA_LOOP_PROGRAMS, bm, bn, device=dev, dtype=torch.float32)
    return lambda: _wmma_loop_kernel[(WMMA_LOOP_PROGRAMS,)](
        a, b, c, iters, BM=bm, BN=bn, BK=bk, UNROLL=WMMA_LOOP_UNROLL, num_warps=warps
    )


def _wmma_loop_candidate(
    bm: int, bn: int, bk: int, warps: int, dtype: torch.dtype, dev: str
) -> _Candidate:
    flop_per_iter = 2 * bm * bn * bk * WMMA_LOOP_PROGRAMS
    iters = max(
        WMMA_LOOP_UNROLL,
        int(WMMA_LOOP_FLOP_PER_LAUNCH / flop_per_iter)
        // WMMA_LOOP_UNROLL
        * WMMA_LOOP_UNROLL,
    )
    return _Candidate(
        {"tile": [bm, bn, bk], "num_warps": warps},
        partial(_wmma_loop_call, bm, bn, bk, warps, iters, dtype, dev),
        flop_per_iter * iters,
    )


def bench_wmma_loop(device: int, proto: _RdnaProtocol) -> Dict[str, Dict[str, Any]]:
    """
    The WMMA rate with no memory traffic: ``acc = tl.dot(a, b, acc)`` on tiles
    held in registers, for BF16 and FP16. Recorded with its clock next to the
    matrix keys, so the gap from a library GEMM splits into clock and kernel;
    it is not a key.
    """
    dev = f"cuda:{device}"
    out: Dict[str, Dict[str, Any]] = {}
    if triton is None:
        return out
    for key, dtype in (("matrix_bf16", torch.bfloat16), ("matrix_fp16", torch.float16)):
        candidates = [
            _wmma_loop_candidate(bm, bn, bk, warps, dtype, dev)
            for bm, bn, bk, warps in WMMA_LOOP_CONFIGS
        ]
        print(f"\n  [WMMA loop, {key}]")
        best, rows = proto.best_of(candidates)
        out[key] = _section(best, rows, "tflops")
    return out


# ── Vector ───────────────────────────────────────────────────────────


def _vector_call(
    kernel: str,
    block: int,
    warps: int,
    dtype: torch.dtype,
    dev: str,
    repeat: int,
    chain_kernel: Any = None,
    chain_args: Optional[Mapping[str, Any]] = None,
):
    n = VECTOR_N_ELEMENTS if kernel == "chain" else VECTOR_N_ELEMENTS // VECTOR_TILE_ILP
    x, y, z = (torch.randn(n, device=dev, dtype=dtype) for _ in range(3))
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block),)
    if kernel == "chain":
        return lambda: chain_kernel[grid](
            x,
            y,
            z,
            out,
            n,
            BLOCK_SIZE=block,
            REPEAT=repeat,
            ACC_FP32=int(dtype == torch.float32),
            **(chain_args or {}),
            num_warps=warps,
        )
    return lambda: _fma_tile_kernel[grid](
        x,
        y,
        z,
        out,
        n,
        BLOCK_SIZE=block,
        ILP=VECTOR_TILE_ILP,
        REPEAT=repeat,
        UNROLL=VECTOR_TILE_UNROLL,
        num_warps=warps,
    )


def _vector_candidates(
    dtype: torch.dtype, dev: str, repeat: int, suite: SuiteParts
) -> List[_Candidate]:
    configs = [("chain", b, w) for b, w in VECTOR_CONFIGS_RDNA] + [
        ("tile", b, w) for b, w in VECTOR_TILE_CONFIGS_RDNA
    ]
    return [
        _Candidate(
            {
                "kernel": kernel,
                "block": block,
                "num_warps": warps,
                **({"ilp": VECTOR_TILE_ILP} if kernel == "tile" else {}),
            },
            partial(
                _vector_call,
                kernel,
                block,
                warps,
                dtype,
                dev,
                repeat,
                suite.chain_kernel,
                suite.chain_args,
            ),
            2 * repeat * VECTOR_N_ELEMENTS,
        )
        for kernel, block, warps in configs
    ]


def bench_vector_rdna(
    device: int, proto: _RdnaProtocol, suite: SuiteParts, repeat: int = 1024
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], List[_Winner]]:
    """
    ``vector_fp16``, ``vector_fp32`` and ``vector_fp64``, each the best of the
    suite's FMA chain at ``VECTOR_CONFIGS_RDNA`` and the accumulator tile at
    ``VECTOR_TILE_CONFIGS_RDNA``; records the share of the winner's FMAs that
    issue as VOPD pairs. ``vector_bf16`` is derived by :func:`rdna_published_keys`.
    """
    dev = f"cuda:{device}"
    results: Dict[str, float] = {}
    sections: Dict[str, Dict[str, Any]] = {}
    winners: List[_Winner] = []
    if triton is None:
        print("  Triton is not available; vector TFLOPS not measured.")
        return results, sections, winners
    for key, dtype in (
        ("vector_fp16", torch.float16),
        ("vector_fp32", torch.float32),
        ("vector_fp64", torch.float64),
    ):
        candidates = _vector_candidates(dtype, dev, repeat, suite)
        print(f"\n  [{key}]  best of {len(candidates)} configs")
        best, rows = proto.best_of(candidates)
        results[key] = round(best["rate"], 1) if best else 0.0
        sections[key] = _section(best, rows, "tflops")
        if best:
            compiled = best["candidate"].factory()()
            torch.cuda.synchronize()
            share = vopd_share(compiled.asm.get("amdgcn", ""))
            sections[key]["vopd_share"] = None if share is None else round(share, 3)
            winners.append(
                _Winner("vector", key, best["candidate"], "tflops", best["rate"])
            )
    return results, sections, winners


# ── Bandwidth ────────────────────────────────────────────────────────


def _read_call(x: torch.Tensor, block: int, grid: int):
    out = torch.empty(block * grid, device=x.device, dtype=torch.float32)
    n = x.numel()
    return lambda: _read_kernel[(grid,)](x, out, n, BLOCK=block, num_warps=8)


def _read_buffer_call(nbytes: int, block: int, grid: int, dev: str):
    return _read_call(
        torch.randn(nbytes // 4, device=dev, dtype=torch.float32), block, grid
    )


def _sum_buffer_call(nbytes: int, dev: str):
    src = torch.randn(nbytes // 4, device=dev, dtype=torch.float32)
    return lambda: src.sum()


def _fastest_read_config(
    nbytes: int, dev: str, flush: Callable[[], Any]
) -> Tuple[int, int]:
    src = torch.randn(nbytes // 4, device=dev, dtype=torch.float32)
    return min(
        BW_READ_CONFIGS,
        key=lambda c: _time_clean(_read_call(src, *c), flush, BW_TUNE_WINDOW_MS),
    )


def _copy_call(nbytes: int, dev: str):
    src = torch.randn(nbytes // 4, device=dev, dtype=torch.float32)
    dst = torch.empty_like(src)
    return lambda: dst.copy_(src)


def _fill_call(nbytes: int, dev: str):
    dst = torch.empty(nbytes // 4, device=dev, dtype=torch.float32)
    return lambda: dst.fill_(1.0)


# Read-only kernels the bandwidth key takes the higher plateau of, by JSON name.
READ_KERNELS: Tuple[Tuple[str, str], ...] = (
    ("triton", "Triton read kernel"),
    ("torch_sum", "torch.sum"),
)


def _read_candidate(
    kernel: str, nbytes: int, dev: str, flush: Callable[[], Any]
) -> _Candidate:
    if kernel == "triton":
        block, grid = _fastest_read_config(nbytes, dev, flush)
        return _Candidate(
            {"kernel": "triton", "bytes": nbytes, "block": block, "grid": grid},
            partial(_read_buffer_call, nbytes, block, grid, dev),
            nbytes,
        )
    return _Candidate(
        {"kernel": "torch.sum", "bytes": nbytes},
        partial(_sum_buffer_call, nbytes, dev),
        nbytes,
    )


def _read_curve(
    kernel: str,
    sizes: List[int],
    dev: str,
    proto: _RdnaProtocol,
    flush: Callable[[], Any],
    *,
    warm: bool,
) -> Tuple[Dict[str, Any], Optional[_Candidate]]:
    """One read-only kernel over *sizes*, each call after the read-only *flush*;
    its plateau (:func:`bandwidth_plateau`), or its largest rate if the curve
    still moves, and the candidate at the size that sets it."""

    def clean_timer(fn):
        t0 = time.time()
        ms = _time_clean(fn, flush, proto.window_ms)
        return ms, proto.clocks(t0, time.time())

    rows: List[Dict[str, Any]] = []
    cands: Dict[int, _Candidate] = {}
    for i, nbytes in enumerate(sizes):
        cand = _read_candidate(kernel, nbytes, dev, flush)
        _, found = proto.best_of(
            [cand], rate_key="gbps", timer=clean_timer, warm=warm and i == 0
        )
        rows += found
        cands[nbytes] = cand
    measured = [r for r in rows if "gbps" in r]
    plateau = bandwidth_plateau(
        [r["bytes"] for r in measured], [r["gbps"] for r in measured]
    )
    if plateau is not None:
        start, gbps = plateau
        rule = f"plateau ({100 * BW_PLATEAU_TOL:g}%) from {start / 2**30:g} GiB, median"
    else:
        start = max(measured, key=lambda r: r["gbps"])["bytes"] if measured else None
        gbps = max((r["gbps"] for r in measured), default=0.0)
        rule = "largest rate (no plateau within the sizes)"
    curve = {
        "gbps": round(gbps, 1),
        "rule": rule,
        "plateau_from_bytes": start,
        "clock": next((_clock_of(r) for r in measured if r["bytes"] == start), {}),
        "rows": rows,
    }
    return curve, cands.get(start)


def _copy_fill_diagnostics(
    dev: str, proto: _RdnaProtocol, free_bytes: int
) -> Dict[str, Dict[str, Any]]:
    """Copy (2n bytes) and fill, timed as the matrix keys: fill after
    ``do_bench``'s cache clear writes into dirty cache lines and can exceed the
    DRAM rate, so neither is published."""
    sizes = [s for s in BW_DIAG_SIZES if 2 * s <= free_bytes // 2]
    out = {}
    for name, call, traffic in (("copy", _copy_call, 2), ("fill", _fill_call, 1)):
        print(f"\n  [{name}, diagnostic]")
        best, rows = proto.best_of(
            [
                _Candidate({"bytes": n}, partial(call, n, dev), traffic * n)
                for n in sizes
            ],
            rate_key="gbps",
            warm=False,
        )
        out[name] = _section(best, rows, "gbps")
    return out


def bench_bandwidth_rdna(
    device: int, proto: _RdnaProtocol
) -> Tuple[float, Dict[str, Any], List[_Winner]]:
    """
    ``mem_bw_gbps`` on an RDNA target: the higher plateau of two read-only
    kernels (:data:`READ_KERNELS`) over ``BW_SIZES_RDNA`` (capped at half the
    free memory), each call after a read-only sweep of ``BW_FLUSH_BYTES`` that
    evicts the caches without leaving dirty lines to write back. Copy and fill
    are recorded as diagnostics (:func:`_copy_fill_diagnostics`).
    """
    dev = f"cuda:{device}"
    if triton is None:
        return 0.0, {"rule": "Triton is not available"}, []
    flush = _read_call(
        torch.ones(BW_FLUSH_BYTES // 4, device=dev, dtype=torch.float32),
        *BW_READ_CONFIGS[1],
    )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    sizes = [s for s in BW_SIZES_RDNA if s <= free_bytes // 2]
    curves: Dict[str, Dict[str, Any]] = {}
    winners: Dict[str, Optional[_Candidate]] = {}
    for i, (name, label) in enumerate(READ_KERNELS):
        print(f"\n  [{label}, read-only flush before each call]")
        curves[name], winners[name] = _read_curve(
            name, sizes, dev, proto, flush, warm=(i == 0)
        )
    best = max(curves, key=lambda name: curves[name]["gbps"])
    mem_bw = curves[best]["gbps"]
    print(f"    mem_bw_gbps = {mem_bw:.1f} GB/s: {best}, {curves[best]['rule']}")
    section = {
        "gbps": mem_bw,
        "kernel": best,
        "rule": curves[best]["rule"],
        "plateau_from_bytes": curves[best]["plateau_from_bytes"],
        "clock": curves[best]["clock"],
        "read": curves,
        **_copy_fill_diagnostics(dev, proto, free_bytes),
    }
    winner = winners[best]
    found = (
        [_Winner("bandwidth", "read", winner, "gbps", mem_bw, flush=flush)]
        if winner is not None
        else []
    )
    return mem_bw, section, found


# ── Side measurements ────────────────────────────────────────────────


def _burst_pass(
    winners: List[_Winner], proto: _RdnaProtocol, idle_s: float, window_ms: int
) -> List[Tuple[_Winner, Dict[str, Any]]]:
    """
    Each section's winner once more, timed over *window_ms* right after *idle_s*
    without GPU work: the rate a short job after an idle gap sees, while the
    clocks are still rising, next to the published one.
    """
    out = []
    for winner in winners:
        cand = winner.candidate
        unit, scale = _RATE_UNITS[winner.rate_key]
        with _preferred_blas(cand.blas):
            fn = cand.factory()
            torch.cuda.synchronize()
            time.sleep(idle_s)
            t0 = time.time()
            if winner.flush is not None:
                ms = _time_clean(fn, winner.flush, window_ms)
            else:
                ms = do_bench(fn, warmup=1, rep=window_ms, return_mode="median")
            clock = proto.clocks(t0, time.time())
        del fn
        rate = cand.work / (ms * 1e-3) / scale
        print(
            f"  [{winner.key}] burst {rate:8.1f} {unit}  ({_config_label(cand.config)})"
        )
        out.append((winner, {winner.rate_key: round(rate, 1), **clock}))
    return out


def _median_or_none(values: List[float]) -> Optional[float]:
    return round(statistics.median(values), 1) if values else None


def _back_to_back(
    winner: _Winner, proto: _RdnaProtocol, seconds: float, settle_s: float
) -> Dict[str, Any]:
    """
    *winner* run back to back for *seconds*, timed in chunks of about
    ``RDNA_BACK_TO_BACK_CHUNK_S``: the median chunk after *settle_s*, by which
    time the board has settled at its power limit, with the clocks and power of
    that part; the median of the chunks before it; and the slowest and fastest
    chunk after it.

    The first chunk has the calls the winner's rate fits in
    ``RDNA_BACK_TO_BACK_FIRST_CHUNK_S``; the rest are sized from its wall time,
    which includes the launch overhead the rate leaves out.
    """
    cand = winner.candidate
    rate_key = winner.rate_key
    unit, scale = _RATE_UNITS[rate_key]
    calls = max(
        1, round(RDNA_BACK_TO_BACK_FIRST_CHUNK_S * winner.rate * scale / cand.work)
    )
    chunks: List[Tuple[float, float]] = []
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    with _preferred_blas(cand.blas):
        fn = cand.factory()
        t0, wall0 = time.perf_counter(), time.time()
        while time.perf_counter() - t0 < seconds:
            started = time.perf_counter()
            begin.record()
            for _ in range(calls):
                fn()
            end.record()
            end.synchronize()
            done = time.perf_counter()
            rate = cand.work * calls / (begin.elapsed_time(end) * 1e-3) / scale
            chunks.append((done - t0, rate))
            if len(chunks) == 1:
                calls = max(
                    1, round(calls * RDNA_BACK_TO_BACK_CHUNK_S / (done - started))
                )
        del fn
    late = [r for t, r in chunks if t > settle_s]
    result = {
        rate_key: _median_or_none(late),
        f"first_{settle_s:g}_s_{rate_key}": _median_or_none(
            [r for t, r in chunks if t <= settle_s]
        ),
        f"min_{rate_key}": round(min(late), 1) if late else None,
        f"max_{rate_key}": round(max(late), 1) if late else None,
        "seconds": seconds,
        "settle_s": settle_s,
        "chunks": len(chunks),
        "calls_per_chunk": calls,
        **proto.clocks(wall0 + settle_s, time.time()),
    }
    print(
        f"  [{winner.key}] back to back {result[rate_key]} {unit} after {settle_s:g} s "
        f"({_config_label(cand.config)})"
    )
    return result


def _side_passes(
    measurement: Dict[str, Any],
    winners: List[_Winner],
    proto: _RdnaProtocol,
    *,
    skip_burst: bool,
    skip_back_to_back: bool,
) -> None:
    """Adds ``burst`` and ``back_to_back`` to the winners' sections of *measurement*."""

    def target(winner: _Winner) -> Dict[str, Any]:
        section = measurement[winner.section]
        return section if winner.section == "bandwidth" else section[winner.key]

    if not skip_burst:
        print(
            f"\n── Burst pass ({RDNA_BURST_WINDOW_MS} ms after {RDNA_BURST_IDLE_S:g} s idle) ──"
        )
        for winner, result in _burst_pass(
            winners, proto, RDNA_BURST_IDLE_S, RDNA_BURST_WINDOW_MS
        ):
            target(winner)["burst"] = result
    if not skip_back_to_back:
        print(f"\n── Back to back ({RDNA_BACK_TO_BACK_S:g} s per winner) ──")
        for winner in winners:
            if winner.section == "matrix" and winner.key in BACK_TO_BACK_KEYS:
                target(winner)["back_to_back"] = _back_to_back(
                    winner, proto, RDNA_BACK_TO_BACK_S, RDNA_BACK_TO_BACK_SETTLE_S
                )


# ── Protocol ─────────────────────────────────────────────────────────


def _protocol_record(
    proto: _RdnaProtocol,
    *,
    skip_burst: bool,
    skip_back_to_back: bool,
    cli_overrides: Sequence[str],
) -> Dict[str, Any]:
    return {
        "warm_s": proto.warm_s,
        "rewarm_s": RDNA_REWARM_S,
        "rewarm_after_s": RDNA_REWARM_AFTER_S,
        "window_ms": proto.window_ms,
        "statistic": "median",
        "do_bench_warmup_ms": DO_BENCH_WARMUP_MS,
        "cli_overrides": list(cli_overrides),
        "burst": (
            None
            if skip_burst
            else {"idle_s": RDNA_BURST_IDLE_S, "window_ms": RDNA_BURST_WINDOW_MS}
        ),
        "back_to_back": (
            None
            if skip_back_to_back
            else {
                "keys": list(BACK_TO_BACK_KEYS),
                "seconds": RDNA_BACK_TO_BACK_S,
                "settle_s": RDNA_BACK_TO_BACK_SETTLE_S,
                "chunk_s": RDNA_BACK_TO_BACK_CHUNK_S,
                "first_chunk_s": RDNA_BACK_TO_BACK_FIRST_CHUNK_S,
            }
        ),
        "telemetry": {
            "source": "sysfs hwmon" if proto.telemetry.available else None,
            "period_s": TELEMETRY_PERIOD_S,
        },
    }


def _platform(device: int, gfx: str, props, bdf: str) -> Dict[str, Any]:
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "gfx_arch": gfx,
        "multi_processor_count": props.multi_processor_count,
        "is_integrated": bool(getattr(props, "is_integrated", False)),
        **platform_record(bdf),
        "hip_version": torch.version.hip,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__ if triton is not None else None,
    }


def _measure_sections(
    device: int,
    gfx: str,
    proto: _RdnaProtocol,
    suite: SuiteParts,
    *,
    repeat: int,
    skip_vector: bool,
    skip_bandwidth: bool,
) -> Tuple[Dict[str, Any], Dict[str, float], float, List[_Winner]]:
    """The matrix, WMMA-loop, vector and bandwidth sections: their records, the
    measured keys, ``mem_bw_gbps`` and the winners."""
    sections: Dict[str, Any] = {}
    print("\n── Matrix TFLOPS (RDNA protocol) ──")
    keys, sections["matrix"], winners = bench_matrix_rdna(device, gfx, proto, suite)
    sections["wmma_loop"] = bench_wmma_loop(device, proto)
    if not skip_vector:
        print("\n── Vector TFLOPS (RDNA protocol) ──")
        vector, sections["vector"], found = bench_vector_rdna(
            device, proto, suite, repeat
        )
        keys.update(vector)
        winners += found
    mem_bw = 0.0
    if not skip_bandwidth:
        print("\n── Memory Bandwidth (RDNA protocol) ──")
        mem_bw, sections["bandwidth"], found = bench_bandwidth_rdna(device, proto)
        winners += found
        if mem_bw <= 0:
            raise MeasurementError(f"{gfx}: read bandwidth measured zero or failed")
    return sections, keys, mem_bw, winners


def run_rdna_protocol(
    device: int,
    gfx: str,
    props,
    suite: SuiteParts,
    *,
    warm_s: float = RDNA_WARM_S,
    window_ms: int = RDNA_WINDOW_MS,
    repeat: int = 1024,
    skip_vector: bool = False,
    skip_bandwidth: bool = False,
    skip_burst: bool = False,
    skip_back_to_back: bool = False,
    cli_overrides: Sequence[str] = (),
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, Any]]:
    """
    Every section on an RDNA target under one protocol (see the module
    docstring). Returns the matrix and vector keys as
    :func:`rdna_published_keys` publishes them, ``mem_bw_gbps``, and the
    top-level JSON extras: ``max_achievable_tflops_source``, ``measurement``
    and ``platform``. Raises :class:`MeasurementError` for a zero or failed
    measurement of a key the target has a path for.
    """
    bdf = pci_bdf(props)
    telemetry = SysfsTelemetry(bdf).start()
    proto = _RdnaProtocol(telemetry, warm_s=warm_s, window_ms=window_ms)
    measurement: Dict[str, Any] = {
        "protocol": _protocol_record(
            proto,
            skip_burst=skip_burst,
            skip_back_to_back=skip_back_to_back,
            cli_overrides=cli_overrides,
        )
    }
    try:
        sections, keys, mem_bw, winners = _measure_sections(
            device,
            gfx,
            proto,
            suite,
            repeat=repeat,
            skip_vector=skip_vector,
            skip_bandwidth=skip_bandwidth,
        )
        measurement.update(sections)
        _side_passes(
            measurement,
            winners,
            proto,
            skip_burst=skip_burst,
            skip_back_to_back=skip_back_to_back,
        )
    finally:
        telemetry.stop()

    rows = {**measurement["matrix"], **measurement.get("vector", {})}
    configs = {k: {"config": s["config"]} for k, s in rows.items() if "config" in s}
    skipped = {"vector_fp16", "vector_fp32", "vector_fp64"} if skip_vector else set()
    values, sources = rdna_published_keys(gfx, keys, configs, skipped)
    platform = _platform(device, gfx, props, bdf)
    platform["power_held_w"] = power_held_w(measurement)
    extras = {
        "max_achievable_tflops_source": sources,
        "measurement": measurement,
        "platform": platform,
    }
    matrix_keys = {k: v for k, v in values.items() if k.startswith("matrix_")}
    vector_keys = {k: v for k, v in values.items() if k.startswith("vector_")}
    return matrix_keys, vector_keys, mem_bw, extras


# ── Command line ─────────────────────────────────────────────────────


def add_arguments(parser) -> None:
    """The protocol's options on :mod:`microbench`'s command line."""
    parser.add_argument(
        "--warm-s",
        type=float,
        default=RDNA_WARM_S,
        help="RDNA targets: seconds of back-to-back calls before each section "
        f"(default: {RDNA_WARM_S:g})",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=RDNA_WINDOW_MS,
        help="RDNA targets: timing window per config in milliseconds, reported "
        f"as the median (default: {RDNA_WINDOW_MS})",
    )
    parser.add_argument(
        "--skip-burst",
        action="store_true",
        help="RDNA targets: skip the burst pass (each winner timed again after "
        f"{RDNA_BURST_IDLE_S:g} s idle)",
    )
    parser.add_argument(
        "--skip-back-to-back",
        action="store_true",
        help="RDNA targets: skip the back-to-back run of the 16-bit GEMM winners "
        f"({RDNA_BACK_TO_BACK_S:g} s each)",
    )


def check_idle(props, util_threshold: int) -> Tuple[bool, str]:
    """The protocol's idle check (:func:`check_rdna_gpu_idle`) on *props*' device;
    VRAM use is not counted on an APU."""
    integrated = bool(getattr(props, "is_integrated", False))
    return check_rdna_gpu_idle(
        pci_bdf(props),
        util_threshold=util_threshold,
        mem_threshold_pct=None if integrated else 10.0,
    )
