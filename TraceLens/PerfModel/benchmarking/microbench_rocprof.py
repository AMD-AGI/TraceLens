#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""microbench_rocprof.py — compare microbench TFLOPS vs rocprofv3 MFMA counters.

For each (metric, shape): one plain run, then one rocprofv3 PMC run wrapping
the same single-shape benchmark. The CSV's last GEMM dispatch row is taken as
steady-state.

Single-run mode (wrapped by rocprofv3):
    python -m TraceLens.PerfModel.benchmarking.microbench_rocprof --single-run \\
        --metric matrix_fp16 --shape 8192,8192,8192 --out /tmp/timing.json

Orchestrator mode (default):
    python -m TraceLens.PerfModel.benchmarking.microbench_rocprof --device 0 \\
        --output results/rocprof_compare.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse all the benchmark helpers from microbench.py so the plain run is
# byte-identical to what microbench.py does.
from . import microbench as mb

# Metric -> rocprofv3 PMC counter (None means: benchmark only, no counter).
COUNTER_MAP: Dict[str, Optional[str]] = {
    "matrix_fp16": "SQ_INSTS_VALU_MFMA_MOPS_F16",
    "matrix_bf16": "SQ_INSTS_VALU_MFMA_MOPS_BF16",
    "matrix_fp32": "SQ_INSTS_VALU_MFMA_MOPS_F32",
    "matrix_int8": "SQ_INSTS_VALU_MFMA_MOPS_I8",
    "matrix_fp8": "SQ_INSTS_VALU_MFMA_MOPS_F8",
    "matrix_fp4": "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
    "matrix_fp6": "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
    "matrix_fp64": "SQ_INSTS_VALU_MFMA_MOPS_F64",
}

DEFAULT_METRICS: List[str] = [k for k in mb.ARCH_TLOPS_KEYS if k.startswith("matrix_")]

# Extra per-metric shapes appended to the default ``mb.GEMM_SHAPES`` list. The
# there on MI355X (matches the shape used in aiter's op_tests).
EXTRA_SHAPES_PER_METRIC: Dict[str, List[Tuple[int, int, int]]] = {
    "matrix_fp4": [(4096, 4096, 4096)],
}

# FLOPs per MFMA MOPS counter increment.
# SQ_INSTS_VALU_MFMA_MOPS_* counts MFMA instructions; each MFMA instruction
# on CDNA performs 512 FMA ops per wavefront (so 512 FLOPs per increment when
# counting MAC=1 flop, or 1024 if counting MAC=2 flops). We use 512 here.
FLOPS_PER_MOPS: float = 512.0

# Substrings used to identify a GEMM/MFMA kernel in the rocprof CSV.
GEMM_NAME_HINTS: Tuple[str, ...] = (
    "Cijk_",
    "gemm",
    "Gemm",
    "GEMM",
    "gett",
    "_scaled_mm",
    "matmul",
    "hgemm",
    "sgemm",
    "igemm",
    "mxfp",
    "_mm_",
)


# ── single-run dispatch (Mode A) ──────────────────────────────────────────


def _run_single(metric: str, M: int, N: int, K: int, device: int) -> Dict[str, float]:
    """Run exactly one (metric, shape) GEMM via the original microbench helpers."""
    import torch  # local import: only needed in single-run path

    torch.cuda.set_device(device)

    flops_per_call = float(mb._gemm_flops(M, N, K))
    extra: Dict[str, float] = {}

    if metric == "matrix_fp16":
        tflops = mb.bench_gemm(M, N, K, torch.float16, device)
    elif metric == "matrix_bf16":
        tflops = mb.bench_gemm(M, N, K, torch.bfloat16, device)
    elif metric == "matrix_fp32":
        tflops = mb.bench_gemm(M, N, K, torch.float32, device)
    elif metric == "matrix_fp64":
        tflops = mb.bench_gemm(M, N, K, torch.float64, device)
    elif metric == "matrix_fp8":
        tflops = mb.bench_gemm_fp8(M, N, K, device)
    elif metric == "matrix_int8":
        # Run torch._int_mm and aiter CK; take max. Rerun winner so it's the
        # last GEMM dispatch in any wrapping rocprof trace.
        torch_tflops = mb.bench_gemm_int8(M, N, K, device)
        try:
            ck_fn = getattr(mb, "bench_int8_ck_gemm", None)
            ck_tflops = (
                ck_fn(
                    M,
                    N,
                    K,
                    device,
                    warmup=mb.WARMUP,
                    rep=mb.REP,
                    do_bench_fn=mb.do_bench,
                )
                if ck_fn is not None
                else 0.0
            )
        except Exception:
            ck_tflops = 0.0
        if ck_tflops > torch_tflops and ck_tflops > 0:
            tflops = mb.bench_int8_ck_gemm(
                M, N, K, device, warmup=mb.WARMUP, rep=mb.REP, do_bench_fn=mb.do_bench
            )
            kernel_kind = "ck"
        else:
            tflops = mb.bench_gemm_int8(M, N, K, device)
            kernel_kind = "torch_int_mm"
        extra["int8_torch_tflops"] = float(torch_tflops)
        extra["int8_ck_tflops"] = float(ck_tflops)
        extra["int8_winner"] = kernel_kind
    elif metric == "matrix_fp4":
        if not mb.mx_available():
            raise RuntimeError("Triton MX not available")
        # Run Triton dot_scaled and aiter CK gemm_a4w4; take max. Rerun winner
        # so it's the last GEMM dispatch in any wrapping rocprof trace.
        tri = mb.bench_mxfp4_gemm(
            M, N, K, device, warmup=mb.WARMUP, rep=mb.REP, do_bench_fn=mb.do_bench
        )
        try:
            ck_fn = getattr(mb, "bench_mxfp4_ck_gemm", None)
            ck = (
                ck_fn(
                    M,
                    N,
                    K,
                    device,
                    warmup=mb.WARMUP,
                    rep=mb.REP,
                    do_bench_fn=mb.do_bench,
                )
                if ck_fn is not None
                else 0.0
            )
        except Exception:
            ck = 0.0
        if ck > tri and ck > 0:
            winner_fn = mb.bench_mxfp4_ck_gemm
            kernel_kind = "ck"
        else:
            winner_fn = mb.bench_mxfp4_gemm
            kernel_kind = "triton"
        tflops = winner_fn(
            M, N, K, device, warmup=mb.WARMUP, rep=mb.REP, do_bench_fn=mb.do_bench
        )
        extra["fp4_triton_tflops"] = float(tri)
        extra["fp4_ck_tflops"] = float(ck)
        extra["fp4_winner"] = kernel_kind
    elif metric == "matrix_fp6":
        if not mb.mx_available():
            raise RuntimeError("Triton MX not available")
        tflops = mb.bench_mxfp6_gemm(
            M, N, K, device, warmup=mb.WARMUP, rep=mb.REP, do_bench_fn=mb.do_bench
        )
    else:
        raise ValueError(f"unknown metric: {metric}")

    measured_ms = (flops_per_call / (tflops * 1e12) * 1e3) if tflops > 0 else 0.0
    result: Dict[str, float] = {
        "metric": metric,
        "M": M,
        "N": N,
        "K": K,
        "measured_ms": float(measured_ms),
        "measured_tflops": float(tflops),
        "flops_per_call": flops_per_call,
        "warmup": mb.WARMUP,
        "rep": mb.REP,
    }
    result.update(extra)
    return result


# ── rocprof CSV parser ────────────────────────────────────────────────────


def _find_pmc_csv(root: Path) -> Optional[Path]:
    """rocprofv3 writes <pid>_counter_collection.csv (sometimes under pmc_*/)."""
    candidates = list(root.rglob("*counter_collection.csv"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1]
    csvs = list(root.rglob("*.csv"))
    csvs.sort(key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _find_col(headers_norm: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
    """Return index of first header matching a candidate (candidate-order priority)."""
    for c in candidates:
        for i, h in enumerate(headers_norm):
            if h == c:
                return i
    for c in candidates:
        for i, h in enumerate(headers_norm):
            if c in h:
                return i
    return None


def _parse_rocprof_csv(csv_path: Path, counter: str) -> Dict[str, object]:
    """Return last-GEMM-row stats: kernel_ms, mops, kernel_name, num_kernels."""
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise RuntimeError(f"empty rocprof CSV: {csv_path}")

    headers = rows[0]
    headers_norm = [_norm(h) for h in headers]

    name_idx = _find_col(headers_norm, ("kernel_name", "kernel"))
    start_idx = _find_col(headers_norm, ("start_timestamp", "start_time", "start"))
    end_idx = _find_col(headers_norm, ("end_timestamp", "end_time", "end"))
    counter_idx = _find_col(headers_norm, (_norm(counter), "counter_value", "value"))

    if start_idx is None or end_idx is None or counter_idx is None:
        raise RuntimeError(f"missing required columns in {csv_path}: headers={headers}")

    def _is_gemm(name: str) -> bool:
        return any(h in name for h in GEMM_NAME_HINTS)

    def _row_vals(r: List[str]) -> Tuple[str, float, float, float]:
        name = r[name_idx] if name_idx is not None else ""
        try:
            start = float(r[start_idx])
            end = float(r[end_idx])
            mops = float(r[counter_idx])
        except (ValueError, IndexError):
            return name, 0.0, 0.0, 0.0
        return name, start, end, mops

    parsed = [_row_vals(r) for r in rows[1:] if r]

    gemm_rows = [t for t in parsed if _is_gemm(t[0])]
    if not gemm_rows:
        gemm_rows = [t for t in parsed if t[3] > 0.0]
    if not gemm_rows:
        raise RuntimeError(f"no GEMM/MFMA rows found in {csv_path} (counter={counter})")

    name, start, end, mops = gemm_rows[-1]
    kernel_ms = (end - start) / 1e6  # rocprof timestamps are in nanoseconds
    return {
        "rocprof_csv": str(csv_path),
        "rocprof_kernel_name": name,
        "rocprof_kernel_ms": float(kernel_ms),
        "rocprof_mops": float(mops),
        "rocprof_num_gemm_kernels": len(gemm_rows),
    }


# ── orchestrator (Mode B) ─────────────────────────────────────────────────


def _shape_tag(M: int, N: int, K: int) -> str:
    return f"{M}x{N}x{K}"


def _run_subprocess_single(
    python: str,
    metric: str,
    M: int,
    N: int,
    K: int,
    device: int,
    warmup: Optional[int],
    rep: Optional[int],
    out_json: Path,
    rocprof_cmd: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    cmd: List[str] = list(rocprof_cmd) if rocprof_cmd else []
    if rocprof_cmd:
        cmd.append("--")
    cmd += [
        python,
        "-m",
        "TraceLens.PerfModel.benchmarking.microbench_rocprof",
        "--single-run",
        "--metric",
        metric,
        "--shape",
        f"{M},{N},{K}",
        "--device",
        str(device),
        "--out",
        str(out_json),
    ]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if rep is not None:
        cmd += ["--rep", str(rep)]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def _parse_shapes(spec: Optional[str]) -> List[Tuple[int, int, int]]:
    if not spec:
        return list(mb.GEMM_SHAPES)
    out: List[Tuple[int, int, int]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        M, N, K = (int(x) for x in chunk.split(","))
        out.append((M, N, K))
    return out


def _shape_ok_for_metric(metric: str, M: int, N: int, K: int) -> Tuple[bool, str]:
    if metric in ("matrix_fp4", "matrix_fp6"):
        if K % mb.MX_BLOCK != 0:
            return False, f"K not divisible by MX_BLOCK={mb.MX_BLOCK}"
        if metric == "matrix_fp4" and K % 2 != 0:
            return False, "K not even (fp4)"
    return True, ""


def run_orchestrator(args: argparse.Namespace) -> int:
    out_root = Path(args.output).resolve()
    out_root.parent.mkdir(parents=True, exist_ok=True)

    rocprof_dir = Path(args.rocprof_dir).resolve()
    rocprof_dir.mkdir(parents=True, exist_ok=True)

    metrics = (
        [m.strip() for m in args.metrics.split(",")]
        if args.metrics
        else DEFAULT_METRICS
    )
    shapes = _parse_shapes(args.shapes)

    if not (mb.mx_available()):
        metrics = [m for m in metrics if m not in ("matrix_fp4", "matrix_fp6")]

    if not args.allow_busy:
        from .microbench_utils import check_gpu_idle

        idle, msg = check_gpu_idle(args.device, util_threshold=args.idle_util_threshold)
        print(f"Idle check: {msg}")
        if not idle:
            print(
                "ERROR: GPU is not idle. Stop other workloads on this device or "
                "rerun with --allow-busy to override.",
                file=sys.stderr,
            )
            sys.exit(2)

    print("=" * 64)
    print("microbench_rocprof.py — measured vs rocprofv3 MFMA counters")
    print("=" * 64)
    print(f"Metrics: {metrics}")
    print(f"Shapes:  {shapes}")
    print(f"Output:  {out_root}")
    print(f"PMC dir: {rocprof_dir}")
    print(f"rocprofv3: {args.rocprofv3}")
    print(f"warmup={args.warmup or mb.WARMUP}  rep={args.rep or mb.REP}")
    print("=" * 64)

    rows: List[Dict[str, object]] = []

    for metric in metrics:
        counter = COUNTER_MAP.get(metric)
        metric_shapes = list(shapes)
        if args.shapes is None:
            for extra in EXTRA_SHAPES_PER_METRIC.get(metric, []):
                if extra not in metric_shapes:
                    metric_shapes.append(extra)
        for M, N, K in metric_shapes:
            ok, why = _shape_ok_for_metric(metric, M, N, K)
            tag = f"{metric} ({M}x{N}x{K})"
            print(f"\n── {tag} ──")
            if not ok:
                print(f"  skipped: {why}")
                rows.append(
                    {
                        "metric": metric,
                        "M": M,
                        "N": N,
                        "K": K,
                        "counter": counter,
                        "error": f"skip:{why}",
                    }
                )
                continue

            # Plain run.
            plain_json = rocprof_dir / f"{metric}_{_shape_tag(M, N, K)}_plain.json"
            print("  [1/2] plain timing run...")
            r = _run_subprocess_single(
                python=sys.executable,
                metric=metric,
                M=M,
                N=N,
                K=K,
                device=args.device,
                warmup=args.warmup,
                rep=args.rep,
                out_json=plain_json,
            )
            if r.returncode != 0 or not plain_json.exists():
                err = (r.stderr or "")[-400:]
                print(f"    FAILED rc={r.returncode}: {err}")
                rows.append(
                    {
                        "metric": metric,
                        "M": M,
                        "N": N,
                        "K": K,
                        "counter": counter,
                        "error": f"plain_run_failed: rc={r.returncode}",
                        "stderr_tail": err,
                    }
                )
                continue
            with open(plain_json) as f:
                plain = json.load(f)
            measured_ms = float(plain.get("measured_ms", 0.0))
            measured_tflops = float(plain.get("measured_tflops", 0.0))
            flops_per_call = float(plain.get("flops_per_call", 0.0))
            print(f"    measured: {measured_ms:.3f} ms  {measured_tflops:.1f} TFLOPS")
            if metric == "matrix_fp4":
                tri = float(plain.get("fp4_triton_tflops", 0.0))
                ck = float(plain.get("fp4_ck_tflops", 0.0))
                winner = plain.get("fp4_winner", "")
                print(f"    fp4:      triton={tri:.1f}  ck={ck:.1f}  winner={winner}")
            elif metric == "matrix_int8":
                tor = float(plain.get("int8_torch_tflops", 0.0))
                ck = float(plain.get("int8_ck_tflops", 0.0))
                winner = plain.get("int8_winner", "")
                print(
                    f"    int8:     torch_int_mm={tor:.1f}  ck={ck:.1f}  winner={winner}"
                )

            # Surface child-process [fp4fp6] diagnostics when a CK path returned 0.
            ck_zero = (
                metric == "matrix_fp4" and float(plain.get("fp4_ck_tflops", 0.0)) == 0.0
            ) or (
                metric == "matrix_int8"
                and float(plain.get("int8_ck_tflops", 0.0)) == 0.0
            )
            if ck_zero:
                child_out = (r.stdout or "") + (r.stderr or "")
                for line in child_out.splitlines():
                    if "[fp4fp6]" in line and (
                        "CK" in line or "INT8" in line or "MXFP" in line
                    ):
                        print(f"    {line}")

            row: Dict[str, object] = {
                "metric": metric,
                "M": M,
                "N": N,
                "K": K,
                "counter": counter,
                "flops_per_call": flops_per_call,
                "measured_ms": measured_ms,
                "measured_tflops": measured_tflops,
            }
            if metric == "matrix_fp4":
                row["fp4_triton_tflops"] = float(plain.get("fp4_triton_tflops", 0.0))
                row["fp4_ck_tflops"] = float(plain.get("fp4_ck_tflops", 0.0))
                row["fp4_winner"] = plain.get("fp4_winner", "")
            elif metric == "matrix_int8":
                row["int8_torch_tflops"] = float(plain.get("int8_torch_tflops", 0.0))
                row["int8_ck_tflops"] = float(plain.get("int8_ck_tflops", 0.0))
                row["int8_winner"] = plain.get("int8_winner", "")

            # rocprof run (skip for metrics with no counter).
            if counter is None:
                print("  [2/2] rocprof: skipped (no counter for this metric)")
                rows.append(row)
                continue

            run_dir = rocprof_dir / f"{metric}_{_shape_tag(M, N, K)}"
            run_dir.mkdir(parents=True, exist_ok=True)
            roc_json = run_dir / "timing.json"
            roc_tag = f"{metric}_{_shape_tag(M, N, K)}"
            rocprof_cmd = [
                args.rocprofv3,
                "--pmc",
                counter,
                "--output-format",
                "csv",
                "-o",
                roc_tag,
                "-d",
                str(run_dir),
            ]
            print(f"  [2/2] rocprofv3 --pmc {counter} ...")
            r = _run_subprocess_single(
                python=sys.executable,
                metric=metric,
                M=M,
                N=N,
                K=K,
                device=args.device,
                warmup=args.warmup,
                rep=args.rep,
                out_json=roc_json,
                rocprof_cmd=rocprof_cmd,
            )
            if r.returncode != 0:
                err = (r.stderr or "")[-400:]
                print(f"    rocprof FAILED rc={r.returncode}: {err}")
                row["error"] = f"rocprof_failed: rc={r.returncode}"
                row["stderr_tail"] = err
                rows.append(row)
                continue

            try:
                pmc_csv = _find_pmc_csv(run_dir)
                if pmc_csv is None:
                    raise RuntimeError(f"no CSV under {run_dir}")
                parsed = _parse_rocprof_csv(pmc_csv, counter)
            except Exception as e:
                print(f"    parse FAILED: {e}")
                row["error"] = f"parse_failed: {e}"
                rows.append(row)
                continue

            row.update(parsed)
            rocprof_kernel_ms = float(parsed["rocprof_kernel_ms"])
            rocprof_mops = float(parsed["rocprof_mops"])
            rocprof_flops = FLOPS_PER_MOPS * rocprof_mops
            rocprof_tflops = (
                rocprof_flops / (rocprof_kernel_ms * 1e-3) / 1e12
                if rocprof_kernel_ms > 0
                else 0.0
            )
            calculated_flops = flops_per_call  # 2*M*N*K from microbench._gemm_flops
            flops_delta = rocprof_flops - calculated_flops
            flops_ratio = (
                rocprof_flops / calculated_flops if calculated_flops > 0 else 0.0
            )
            row["calculated_flops"] = calculated_flops
            row["rocprof_flops"] = rocprof_flops
            row["rocprof_tflops"] = rocprof_tflops
            row["flops_delta_rocprof_minus_calculated"] = flops_delta
            row["flops_ratio_rocprof_over_calculated"] = flops_ratio
            row["ms_ratio_rocprof_over_measured"] = (
                rocprof_kernel_ms / measured_ms if measured_ms > 0 else 0.0
            )

            print(
                f"    rocprof: {rocprof_kernel_ms:.3f} ms  "
                f"{rocprof_tflops:.1f} TFLOPS  "
                f"(MOPS={rocprof_mops:.3e}, kernels={parsed['rocprof_num_gemm_kernels']})"
            )
            kname = str(parsed.get("rocprof_kernel_name", ""))
            kshort = kname if len(kname) <= 100 else kname[:100] + "..."
            print(f"    kernel:  {kshort}")
            print(
                f"    flops:   calc={calculated_flops:.3e}  "
                f"rocprof={rocprof_flops:.3e}  "
                f"ratio(roc/calc)={flops_ratio:.3f}  "
                f"delta={flops_delta:+.3e}"
            )
            rows.append(row)

    # ── write outputs ──────────────────────────────────────────────────
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warmup": args.warmup or mb.WARMUP,
        "rep": args.rep or mb.REP,
        "counter_map": COUNTER_MAP,
        "rows": rows,
    }
    with open(out_root, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_root}")

    csv_out = out_root.with_suffix(".csv")
    fieldnames = [
        "metric",
        "M",
        "N",
        "K",
        "counter",
        "measured_ms",
        "measured_tflops",
        "calculated_flops",
        "flops_per_call",
        "rocprof_kernel_ms",
        "rocprof_mops",
        "rocprof_flops",
        "rocprof_tflops",
        "rocprof_num_gemm_kernels",
        "rocprof_kernel_name",
        "ms_ratio_rocprof_over_measured",
        "flops_ratio_rocprof_over_calculated",
        "flops_delta_rocprof_minus_calculated",
        "fp4_triton_tflops",
        "fp4_ck_tflops",
        "fp4_winner",
        "int8_torch_tflops",
        "int8_ck_tflops",
        "int8_winner",
        "error",
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_out}")

    # ── summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 130)
    print("SUMMARY  (measured vs rocprofv3)")
    print("=" * 130)
    print(
        f"{'metric':<14} {'shape':<22} "
        f"{'meas ms':>9} {'meas TF':>9} "
        f"{'roc ms':>9} {'roc TF':>9} "
        f"{'calc FLOPs':>12} {'roc FLOPs':>12} "
        f"{'roc/calc':>9} {'ms r':>7}  kernel"
    )
    print("-" * 130)
    for r in rows:
        shape = f"{r.get('M')}x{r.get('N')}x{r.get('K')}"
        meas_ms = float(r.get("measured_ms", 0.0))
        meas_tf = float(r.get("measured_tflops", 0.0))
        roc_ms = float(r.get("rocprof_kernel_ms", 0.0))
        roc_tf = float(r.get("rocprof_tflops", 0.0))
        calc_fl = float(r.get("calculated_flops", 0.0))
        roc_fl = float(r.get("rocprof_flops", 0.0))
        fl_r = float(r.get("flops_ratio_rocprof_over_calculated", 0.0))
        ms_r = float(r.get("ms_ratio_rocprof_over_measured", 0.0))
        err = r.get("error", "")
        kname = str(r.get("rocprof_kernel_name", ""))
        kshort = kname if len(kname) <= 60 else kname[:60] + "..."
        print(
            f"{r['metric']:<14} {shape:<22} "
            f"{meas_ms:>9.3f} {meas_tf:>9.1f} "
            f"{roc_ms:>9.3f} {roc_tf:>9.1f} "
            f"{calc_fl:>12.3e} {roc_fl:>12.3e} "
            f"{fl_r:>9.3f} {ms_r:>7.3f}  {kshort}" + (f"  [{err}]" if err else "")
        )
    print("=" * 130)
    return 0


# ── main / CLI ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="microbench vs rocprofv3 MFMA counter comparison"
    )
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--metrics",
        type=str,
        default=None,
        help=f"Comma list (default: {','.join(DEFAULT_METRICS)})",
    )
    p.add_argument(
        "--shapes",
        type=str,
        default=None,
        help='Semicolon-separated "M,N,K" triples (default: microbench.GEMM_SHAPES)',
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override microbench.WARMUP for both runs",
    )
    p.add_argument(
        "--rep", type=int, default=None, help="Override microbench.REP for both runs"
    )
    p.add_argument(
        "--allow-busy",
        action="store_true",
        help="Skip the pre-flight idle check and run even if the GPU is busy",
    )
    p.add_argument(
        "--idle-util-threshold",
        type=int,
        default=5,
        help="Max GPU utilization %% considered idle for the pre-flight check (default: 5)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "results" / "rocprof_compare.json"
        ),
    )
    p.add_argument(
        "--rocprof-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "rocprof_compare"),
    )
    p.add_argument("--rocprofv3", type=str, default="rocprofv3")

    # Single-run (Mode A) options:
    p.add_argument(
        "--single-run",
        action="store_true",
        help="Run ONE (metric, shape) GEMM and write timing JSON (used as rocprof target)",
    )
    p.add_argument("--metric", type=str, default=None)
    p.add_argument("--shape", type=str, default=None, help='"M,N,K"')
    p.add_argument(
        "--out", type=str, default=None, help="Single-run timing JSON output path"
    )

    args = p.parse_args()

    if args.warmup is not None:
        mb.WARMUP = args.warmup
    if args.rep is not None:
        mb.REP = args.rep

    if args.single_run:
        if not (args.metric and args.shape and args.out):
            print("--single-run requires --metric, --shape, --out", file=sys.stderr)
            return 2
        M, N, K = (int(x) for x in args.shape.split(","))
        result = _run_single(args.metric, M, N, K, args.device)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return 0

    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
