###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Perf-model validation script.

Runs the ``replay/_runner.py`` GPU harness for a registered op under rocprofv3
with hardware counters, then compares measured ops/bytes against the perf
model's predictions. Ops, their perf models and their trace names are
registered in ``harnesses.py``; this script only orchestrates
profiling and reporting.

Usage (single op):
    python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator \
        --op gemm_a8w8_blockscale \
        --M 2048 --N 4096 --K 8192 \
        --arch gfx942 --output-dir /tmp/perf_validation

Usage (batch):
    python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator \
        --all --arch gfx942 --output-dir /tmp/perf_val
    python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator \
        --category activation --arch gfx942 --output-dir /tmp/perf_val

Usage (traced shapes; build the cases file with ModelBringup's bringup_gates cases):
    python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator \
        --cases cases.json --arch gfx942 --output-dir /tmp/perf_val

Usage (discover missing coverage):
    python3 -m TraceLens.Agent.PerfModel.validate_perf_model.validator --discover
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

from TraceLens.Agent.PerfModel.validate_perf_model.harnesses import (
    OP_REGISTRY,
    TRACE_NAME_TO_OP,
    chunk_stats_annotation,
    model_source,
    run_perf_model_from_event,
    tuple_to_list,
)
from TraceLens.Agent.PerfModel.validate_perf_model.ratio_bands import (
    BAND_COLUMNS,
    RATIO_MATCH,
    match_status,
    ratio_band,
)
from TraceLens.Agent.PerfModel.validate_perf_model.replay._runner import (
    INT_FLAGS,
    STR_FLAGS,
)
from TraceLens.Agent.PerfModel.validate_perf_model.replay.attention import (
    compute_varlen_annotation_stats,
    make_varlen_seqlens,
)

# Constants

# Read bytes are reconstructed from the raw TCC->EA read-request counters so
# that 32B, 64B and 128B requests are each weighted by their own size (see
# measured_bytes). Write bytes come from WRITE_SIZE (KiB units).
ARCH_COUNTER_CONFIGS = {
    "gfx942": {
        "ops_counters": ["SQ_INSTS_VALU_MFMA_MOPS_I8", "SQ_INSTS_VALU_MFMA_MOPS_F8"],
        "ops_derived_16": ["TOTAL_16_OPS"],
        "ops_derived_32": ["TOTAL_32_OPS"],
        "mem_read_counters": [
            "TCC_EA0_RDREQ_sum",
            "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA0_RDREQ_64B_sum",
            "TCC_BUBBLE_sum",
        ],
        "mem_write_counters": ["WRITE_SIZE"],
    },
    "gfx950": {
        "ops_counters": [
            "SQ_INSTS_VALU_MFMA_MOPS_I8",
            "SQ_INSTS_VALU_MFMA_MOPS_F8",
            "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
        ],
        "ops_derived_16": ["TOTAL_16_OPS"],
        "ops_derived_32": ["TOTAL_32_OPS"],
        "mem_read_counters": [
            "TCC_EA0_RDREQ_sum",
            "TCC_EA0_RDREQ_32B_sum",
            "TCC_EA0_RDREQ_64B_sum",
            "TCC_BUBBLE_sum",
        ],
        "mem_write_counters": ["WRITE_SIZE"],
    },
}

PRECISION_TO_COUNTERS = {
    "fp8": ["SQ_INSTS_VALU_MFMA_MOPS_F8"],
    "fp4": ["SQ_INSTS_VALU_MFMA_MOPS_F6F4"],
    "int8": ["SQ_INSTS_VALU_MFMA_MOPS_I8"],
    "fp16": ["TOTAL_16_OPS"],
    "bf16": ["TOTAL_16_OPS"],
    "fp32": ["TOTAL_32_OPS"],
}

COUNTER_DISPLAY = {
    "SQ_INSTS_VALU_MFMA_MOPS_I8": (512, "INT8 MFMA Ops", "ops"),
    "SQ_INSTS_VALU_MFMA_MOPS_F8": (512, "FP8 MFMA Ops", "ops"),
    "TOTAL_16_OPS": (1, "Total 16-bit Ops", "ops"),
    "TOTAL_32_OPS": (1, "Total 32-bit Ops", "ops"),
    "SQ_INSTS_VALU_MFMA_MOPS_F6F4": (512, "FP4/FP6 MFMA Ops", "ops"),
    "WRITE_SIZE": (1024, "HBM Write", "bytes"),
}

# Raw TCC read-request counters kept by parse_counter_csv even though they are
# not part of COUNTER_DISPLAY; measured_bytes reconstructs read bytes from them.
_MEM_READ_RAW_COUNTERS = {
    "TCC_EA0_RDREQ_sum",
    "TCC_EA0_RDREQ_32B_sum",
    "TCC_EA0_RDREQ_64B_sum",
    "TCC_BUBBLE_sum",
}

CATEGORIES = sorted({entry["category"] for entry in OP_REGISTRY.values()})

REPLAY_RUNNER = Path(__file__).resolve().parent / "replay" / "_runner.py"
# Kernel replay/_runner.measure() launches right before the measured call.
MARKER_KERNEL = "bitwise_not"

_INFRA_KERNEL_PATTERNS = [
    "at::native::",
    "at::cuda::",
    "void hip",
    "Cijk_",
    "void at::native",
    # Runtime allocator/transfer kernels. Ops that allocate or zero their own
    # buffers (e.g. the vLLM GEMM wrappers) emit many of these, and they can
    # otherwise outweigh the kernel under test during auto-discovery.
    "__amd_rocclr_fillBuffer",
    "__amd_rocclr_copyBuffer",
    "__amd_rocclr_memset",
    "hipMemset",
]

CSV_COLUMNS = [
    "op",
    "trace_name",
    "perf_model_class",
    "source_file",
    "source_line",
    "predicted_precision",
    "predicted_flops",
    "predicted_bytes",
    "csv_reported_flops",
    "csv_reported_bytes",
    "rocprof_f4_ops",
    "rocprof_f8_ops",
    "rocprof_total_16",
    "rocprof_total_32",
    "rocprof_read_bytes",
    "rocprof_write_bytes",
    "kernel_name",
    "kernel_runtime_us",
    "kernel_runtime_mean_us",
    "kernel_runtime_dispatches",
    "throughput_ops_counter",
    "predicted_tflops_s",
    "measured_tflops_s",
    "predicted_tb_s",
    "measured_tb_s",
    "flops_ratio",
    "bytes_ratio",
    *BAND_COLUMNS.values(),
    "run_status",
    "match_status",
    "comment",
]


def _rocprof_env(out_dir):
    """Environment for a rocprofv3 launch whose working directory is ``out_dir``.

    ``cwd=`` alone leaves ``PWD`` at the caller's directory (the op worktree),
    and rocprofv3 writes its ``.rocprofv3/`` scratch directory there.
    """
    return {**os.environ, "PWD": out_dir}


def _find_csv(path, search_dir, suffix):
    """``path`` if it exists, else the first ``*suffix`` file under ``search_dir``."""
    if os.path.exists(path):
        return path
    candidates = list(Path(search_dir).rglob(f"*{suffix}"))
    return str(candidates[0]) if candidates else None


def run_rocprofv3(
    rocprofv3_path, rocprof_args, csv_suffix, runner_argv, out_name, out_dir, timeout
):
    """Run ``runner_argv`` under rocprofv3 and return the CSV it writes.

    ``rocprof_args`` select the collection mode and the CSV is the file ending
    in ``csv_suffix``. Raises RuntimeError on a timeout or a non-zero exit, and
    FileNotFoundError when no such CSV is written.
    """
    out_dir = os.path.abspath(out_dir)
    result_dir = os.path.join(out_dir, out_name)
    os.makedirs(result_dir, exist_ok=True)
    cmd = [
        rocprofv3_path,
        *rocprof_args,
        "--output-format",
        "csv",
        "-o",
        out_name,
        "-d",
        result_dir,
        "--",
        *runner_argv,
    ]
    stdout_log = os.path.join(out_dir, f"{out_name}_stdout.log")
    stderr_log = os.path.join(out_dir, f"{out_name}_stderr.log")
    print(f'\n{"============================================================"}')
    print(f'Running: {" ".join(cmd)}')
    print(f"Logs:    {stdout_log}\n         {stderr_log}")
    print(f"Timeout: {timeout}s")
    print(f'{"============================================================"}')
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=out_dir,
        env=_rocprof_env(out_dir),
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    Path(stdout_log).write_text(stdout or "")
    Path(stderr_log).write_text(stderr or "")
    if timed_out:
        raise RuntimeError(
            f"rocprofv3 timed out after {timeout}s. Check logs:\n  {stdout_log}\n  {stderr_log}"
        )
    if stdout:
        print(f"[{out_name} stdout]:\n{stdout}")
    if stderr:
        print(f"[{out_name} stderr]:\n{stderr}")
    if proc.returncode != 0:
        raise RuntimeError(
            f"rocprofv3 failed with exit code {proc.returncode}. Check logs:\n  {stdout_log}\n  {stderr_log}"
        )
    csv_path = _find_csv(
        os.path.join(result_dir, f"{out_name}_{csv_suffix}"), result_dir, csv_suffix
    )
    if csv_path is None:
        raise FileNotFoundError(
            f"No *{csv_suffix} in {result_dir}. Directory contents: "
            f"{os.listdir(result_dir)}. Check logs:\n  {stdout_log}\n  {stderr_log}"
        )
    return csv_path


def _is_infra(name):
    return any(pat in name for pat in _INFRA_KERNEL_PATTERNS)


def _measured_dispatches(rows, kernel_filter=None):
    """Rows of the measured call: the dispatches after the last marker kernel.

    ``rows`` are rocprofv3 CSV rows (kernel trace or counter collection).
    Infrastructure kernels are left out, and ``kernel_filter`` (a substring of
    the kernel name) narrows the rest. Raises RuntimeError when the trace has no
    marker.
    """
    markers = [int(r["Dispatch_Id"]) for r in rows if MARKER_KERNEL in r["Kernel_Name"]]
    if not markers:
        raise RuntimeError(f"no {MARKER_KERNEL} marker in trace")
    return [
        r
        for r in rows
        if int(r["Dispatch_Id"]) > max(markers)
        and not _is_infra(r["Kernel_Name"])
        and (not kernel_filter or kernel_filter in r["Kernel_Name"])
    ]


def _read_csv_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def discover_measured_kernels(
    rocprofv3_path, runner_argv, out_dir, timeout=120, kernel_filter=None
):
    """Trace one run of ``runner_argv``; return ``(trace_csv, kernel_names)``.

    ``kernel_names`` are the distinct kernels of the measured call, in launch
    order.
    """
    trace_csv = run_rocprofv3(
        rocprofv3_path,
        ["--kernel-trace"],
        "kernel_trace.csv",
        runner_argv,
        "kernel_discovery",
        out_dir,
        timeout,
    )
    window = _measured_dispatches(_read_csv_rows(trace_csv), kernel_filter)
    names = list(dict.fromkeys(r["Kernel_Name"] for r in window))
    if not names:
        raise RuntimeError("no kernel after the marker in trace")
    print(f"  Measured call: {len(window)} dispatches of {len(names)} kernels:")
    for name in names:
        print(f"    {name[:200]}")
    return trace_csv, names


def parse_kernel_runtime_ns(trace_csv, kernel_filter=None):
    """Return ``(total_ns, mean_ns, num_dispatches)`` for the measured call.

    Reads a rocprofv3 ``--kernel-trace`` CSV and sums the durations of the
    dispatches after the marker.
    """
    durations = [
        int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        for r in _measured_dispatches(_read_csv_rows(trace_csv), kernel_filter)
    ]
    if not durations:
        return (None, None, 0)
    return (sum(durations), sum(durations) / len(durations), len(durations))


def parse_counter_csv(csv_path, kernel_filter=None):
    """Sum each counter over the measured call's dispatches in a ``--pmc`` CSV."""
    window = _measured_dispatches(_read_csv_rows(csv_path), kernel_filter)
    counters = {}
    for row in window:
        name = row["Counter_Name"]
        # Keep counters with a known display mapping, plus the raw TCC
        # read-request counters used for the corrected read-byte formula.
        if name in COUNTER_DISPLAY or name in _MEM_READ_RAW_COUNTERS:
            counters[name] = counters.get(name, 0.0) + float(row["Counter_Value"])
    dispatches = len({r["Dispatch_Id"] for r in window})
    print(f"  Parsed {csv_path}: {dispatches} measured dispatches")
    print(f"  Counters: {counters}")
    return counters


def format_number(val, unit):
    if val is None:
        return "N/A"
    if unit == "bytes":
        if val >= 1e9:
            return f"{val / 1e9:.2f} GB"
        if val >= 1e6:
            return f"{val / 1e6:.2f} MB"
        if val >= 1000:
            return f"{val / 1000:.2f} KB"
        return f"{val:.0f} B"
    if val >= 1e12:
        return f"{val / 1e12:.2f} T"
    if val >= 1e9:
        return f"{val / 1e9:.2f} G"
    if val >= 1e6:
        return f"{val / 1e6:.2f} M"
    if val >= 1000:
        return f"{val / 1000:.2f} K"
    return f"{val:.0f}"


def _arch_ops_counters(arch):
    """Ops counters ``arch`` collects, or an empty set for an unknown arch."""
    cfg = ARCH_COUNTER_CONFIGS.get(arch, {})
    return {
        name
        for key in ("ops_counters", "ops_derived_16", "ops_derived_32")
        for name in cfg.get(key, [])
    }


def measured_bytes(mem_counters):
    """Return ``(read_bytes, write_bytes)`` from the raw memory counters."""
    # TCC_EA0_RDREQ_sum counts all requests; those not tagged 32B or 64B are
    # 128B.
    rd_total = mem_counters.get("TCC_EA0_RDREQ_sum", 0)
    rd32 = mem_counters.get("TCC_EA0_RDREQ_32B_sum", 0)
    rd64 = mem_counters.get("TCC_EA0_RDREQ_64B_sum", 0)
    read_bytes = 32 * rd32 + 64 * rd64 + 128 * (rd_total - rd32 - rd64)
    return read_bytes, mem_counters.get("WRITE_SIZE", 0) * 1024


def generate_report(
    op_name,
    arch,
    dims_str,
    predicted_flops,
    predicted_bytes,
    predicted_precision,
    ops_counters,
    mem_counters,
    output_dir,
    kernel_filter=None,
    kernel_runtime_ns=None,
    kernel_runtime_mean_ns=None,
    kernel_runtime_dispatches=0,
):
    """Generate the validation report.

    Returns a dict with flops_ratio, bytes_ratio, precision, and all
    individual scaled counter values for the rich CSV/text report.
    """
    lines = []
    lines.append(
        "======================================================================"
    )
    lines.append("PERF MODEL VALIDATION REPORT")
    lines.append(
        "======================================================================"
    )
    lines.append(f"Op:   {op_name}")
    lines.append(f"Arch: {arch}")
    lines.append(dims_str)
    if kernel_filter:
        lines.append(f"Kernel: {kernel_filter[:120]}")
    lines.append("")
    lines.append("--- Perf Model Predictions ---")
    lines.append(f'  FLOPs:               {format_number(predicted_flops, "ops")}')
    lines.append(f'  Bytes (roofline min): {format_number(predicted_bytes, "bytes")}')
    lines.append(f'  Compute precision:   {predicted_precision or "unknown"}')
    lines.append("")
    lines.append("--- Measured Ops (rocprofv3 --pmc) ---")
    for counter_name, raw_val in ops_counters.items():
        multiplier, label, unit = COUNTER_DISPLAY[counter_name]
        total = raw_val * multiplier
        lines.append(
            f"  {label:25s}: {format_number(total, unit):>12s}  (raw counter: {raw_val:.0f})"
        )
    lines.append("")
    lines.append("--- Measured Memory (rocprofv3 --pmc) ---")
    measured_read_bytes, measured_write_bytes = measured_bytes(mem_counters)
    rd_total = mem_counters.get("TCC_EA0_RDREQ_sum", 0)
    write_raw = mem_counters.get("WRITE_SIZE", 0)
    lines.append(
        f'  {"HBM Read":25s}: {format_number(measured_read_bytes, "bytes"):>12s}  (raw counter: {rd_total:.0f})'
    )
    lines.append(
        f'  {"HBM Write":25s}: {format_number(measured_write_bytes, "bytes"):>12s}  (raw counter: {write_raw:.0f})'
    )
    measured_total_bytes = measured_read_bytes + measured_write_bytes
    lines.append(
        f'  {"Total HBM Traffic":25s}: {format_number(measured_total_bytes, "bytes"):>12s}'
    )
    lines.append("")
    lines.append("--- Comparison ---")
    lines.append(
        f'  {"Metric":30s} {"Predicted":>14s} {"Measured":>14s} {"Ratio":>10s}'
    )
    lines.append(
        f'  {"------------------------------"} {"--------------"} {"--------------"} {"----------"}'
    )
    flops_ratio = None
    i8_ops = ops_counters.get("SQ_INSTS_VALU_MFMA_MOPS_I8", 0)
    f8_ops = ops_counters.get("SQ_INSTS_VALU_MFMA_MOPS_F8", 0)
    f4_ops = ops_counters.get("SQ_INSTS_VALU_MFMA_MOPS_F6F4", 0)
    t16_ops = ops_counters.get("TOTAL_16_OPS", 0)
    t32_ops = ops_counters.get("TOTAL_32_OPS", 0)

    def _actual_ops(counter_name, raw_val):
        return raw_val * COUNTER_DISPLAY.get(counter_name, (1, "", ""))[0]

    best_counter = max(
        [
            ("SQ_INSTS_VALU_MFMA_MOPS_F8", f8_ops),
            ("SQ_INSTS_VALU_MFMA_MOPS_I8", i8_ops),
            ("SQ_INSTS_VALU_MFMA_MOPS_F6F4", f4_ops),
            ("TOTAL_16_OPS", t16_ops),
            ("TOTAL_32_OPS", t32_ops),
        ],
        key=lambda x: _actual_ops(x[0], x[1]),
    )
    relevant_ops_counter = best_counter[0] if best_counter[1] > 0 else None
    if relevant_ops_counter and predicted_flops and predicted_flops > 0:
        multiplier, label, unit = COUNTER_DISPLAY[relevant_ops_counter]
        measured_ops = ops_counters[relevant_ops_counter] * multiplier
        flops_ratio = measured_ops / predicted_flops
        lines.append(
            f'  {label:30s} {format_number(predicted_flops, "ops"):>14s} '
            f'{format_number(measured_ops, "ops"):>14s} {flops_ratio:>10.4f}'
        )
    bytes_ratio = None
    if predicted_bytes and predicted_bytes > 0 and measured_total_bytes > 0:
        bytes_ratio = measured_total_bytes / predicted_bytes
        lines.append(
            f'  {"Total HBM Bytes":30s} {format_number(predicted_bytes, "bytes"):>14s} '
            f'{format_number(measured_total_bytes, "bytes"):>14s} {bytes_ratio:>10.4f}'
        )
    precision_result = "N/A"
    precision_reason = None
    lines.append("")
    lines.append("--- Compute Precision Check ---")
    if predicted_precision and predicted_precision in PRECISION_TO_COUNTERS:
        expected_counters = PRECISION_TO_COUNTERS[predicted_precision]
        expected_label = ", ".join(expected_counters)
        mfma_candidates = {
            "SQ_INSTS_VALU_MFMA_MOPS_F8",
            "SQ_INSTS_VALU_MFMA_MOPS_I8",
            "SQ_INSTS_VALU_MFMA_MOPS_F6F4",
        }
        all_ops = dict(ops_counters)
        dominant_counter = None
        dominant_val = 0
        for cname, cval in all_ops.items():
            if cname in mfma_candidates and cval > dominant_val:
                dominant_counter = cname
                dominant_val = cval
        if dominant_counter is None and all_ops:
            for cname, cval in all_ops.items():
                if cval > dominant_val:
                    dominant_counter = cname
                    dominant_val = cval
        lines.append(f"  Predicted precision:  {predicted_precision}")
        lines.append(f"  Expected counter(s):  {expected_label}")
        if dominant_counter:
            mult, label, _ = COUNTER_DISPLAY.get(
                dominant_counter, (1, dominant_counter, "ops")
            )
            lines.append(
                f'  Dominant counter:     {dominant_counter}  ({format_number(dominant_val * mult, "ops")})'
            )
            if dominant_counter in expected_counters:
                precision_result = "PASS"
                lines.append(
                    "  Result:               PASS - dominant MFMA type matches predicted precision"
                )
            else:
                precision_result = "MISMATCH"
                lines.append(
                    f"  Result:               MISMATCH - expected {expected_label} but dominant is {dominant_counter}"
                )
                arch_counters = _arch_ops_counters(arch)
                if arch_counters and not set(expected_counters) & arch_counters:
                    precision_reason = (
                        f"{predicted_precision} counter ({expected_label}) "
                        f"is not available on {arch}"
                    )
                    lines.append(f"  Reason:               {precision_reason}")
                lines.append("  All ops counters:")
                for cname, cval in sorted(all_ops.items()):
                    if cval > 0:
                        m, l, _ = COUNTER_DISPLAY.get(cname, (1, cname, "ops"))
                        lines.append(
                            f'    {l:25s}: {format_number(cval * m, "ops"):>12s}'
                        )
        else:
            precision_result = "NO DATA"
            lines.append("  Result:               NO DATA - no ops counters collected")
    elif predicted_precision:
        lines.append(f"  Predicted precision:  {predicted_precision}")
        lines.append(
            f"  Result:               SKIPPED - no counter mapping for '{predicted_precision}'"
        )
    else:
        lines.append(
            "  Result:               SKIPPED - perf model returned no compute precision"
        )
    measured_throughput_ops = None
    measured_throughput_label = None
    if relevant_ops_counter:
        mult = COUNTER_DISPLAY[relevant_ops_counter][0]
        measured_throughput_label = COUNTER_DISPLAY[relevant_ops_counter][1]
        measured_throughput_ops = ops_counters[relevant_ops_counter] * mult
    pred_tflops_s = None
    meas_tflops_s = None
    pred_tb_s = None
    meas_tb_s = None
    runtime_us = None
    runtime_mean_us = None
    if kernel_runtime_ns and kernel_runtime_ns > 0:
        runtime_s = kernel_runtime_ns / 1e9
        runtime_us = kernel_runtime_ns / 1000
        if kernel_runtime_mean_ns:
            runtime_mean_us = kernel_runtime_mean_ns / 1000
        if predicted_flops:
            pred_tflops_s = predicted_flops / 1e12 / runtime_s
        if predicted_bytes:
            pred_tb_s = predicted_bytes / 1e12 / runtime_s
        if measured_throughput_ops:
            meas_tflops_s = measured_throughput_ops / 1e12 / runtime_s
        if measured_total_bytes:
            meas_tb_s = measured_total_bytes / 1e12 / runtime_s
        lines.append("")
        lines.append(
            "--- Achieved Throughput (measured-call runtime from kernel-trace) ---"
        )
        runtime_line = f"  Kernel runtime (measured call): {runtime_us:9.3f} us"
        if runtime_mean_us is not None and kernel_runtime_dispatches > 1:
            runtime_line += f"   (mean over {kernel_runtime_dispatches} dispatches: {runtime_mean_us:.3f} us)"
        lines.append(runtime_line)
        lines.append(f'  {"Metric":32s} {"Predicted":>14s} {"Measured":>14s}')
        lines.append(
            f'  {"--------------------------------"} {"--------------"} {"--------------"}'
        )
        pf_s = (
            f"{pred_tflops_s:>10.3f} TF/s"
            if pred_tflops_s is not None
            else "       N/A"
        )
        mf_s = (
            f"{meas_tflops_s:>10.3f} TF/s"
            if meas_tflops_s is not None
            else "       N/A"
        )
        meas_label = (
            f"Compute (rocprof {measured_throughput_label})"
            if measured_throughput_label
            else "Compute (rocprof)"
        )
        lines.append(f"  {meas_label:32s} {pf_s:>14s} {mf_s:>14s}")
        pb_s = f"{pred_tb_s:>10.3f} TB/s" if pred_tb_s is not None else "       N/A"
        mb_s = f"{meas_tb_s:>10.3f} TB/s" if meas_tb_s is not None else "       N/A"
        lines.append(f'  {"HBM (rocprof RDREQ+WRITE)":32s} {pb_s:>14s} {mb_s:>14s}')
        if pred_tflops_s and pred_tb_s and pred_tb_s > 0:
            lines.append(
                f"  Arithmetic intensity (predicted): {predicted_flops / max(predicted_bytes, 1):.2f} FLOP/byte"
            )
        if meas_tflops_s and meas_tb_s and meas_tb_s > 0:
            ai_meas = (
                measured_throughput_ops / max(measured_total_bytes, 1)
                if measured_throughput_ops
                else 0
            )
            lines.append(f"  Arithmetic intensity (measured):  {ai_meas:.2f} FLOP/byte")
    else:
        lines.append("")
        lines.append("--- Achieved Throughput ---")
        lines.append(
            "  Kernel runtime not available (no kernel-trace CSV); throughput skipped."
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Ratio = Measured / Predicted.  1.0 = perfect match.")
    lines.append(
        "  - Ops ratio < 1.0 may indicate the kernel uses a different MFMA dtype"
    )
    lines.append("    than assumed. Check other ops counters above.")
    lines.append("  - Bytes ratio > 1.0 is expected due to cache-line granularity,")
    lines.append("    L2 replays, scale tensor traffic, and alignment padding.")
    lines.append(
        "======================================================================"
    )
    report_text = "\n".join(lines)
    print(report_text)
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    mfma_f8 = ops_counters.get("SQ_INSTS_VALU_MFMA_MOPS_F8", 0) + ops_counters.get(
        "SQ_INSTS_VALU_MFMA_MOPS_I8", 0
    )
    return {
        "flops_ratio": flops_ratio,
        "bytes_ratio": bytes_ratio,
        "precision": precision_result,
        "precision_reason": precision_reason,
        "rocprof_f4_ops": f4_ops
        * COUNTER_DISPLAY.get("SQ_INSTS_VALU_MFMA_MOPS_F6F4", (1,))[0],
        "rocprof_f8_ops": mfma_f8 * 512,
        "rocprof_total_16": t16_ops * COUNTER_DISPLAY.get("TOTAL_16_OPS", (1,))[0],
        "rocprof_total_32": t32_ops * COUNTER_DISPLAY.get("TOTAL_32_OPS", (1,))[0],
        "rocprof_read_bytes": measured_read_bytes,
        "rocprof_write_bytes": measured_write_bytes,
        "kernel_runtime_us": runtime_us,
        "kernel_runtime_mean_us": runtime_mean_us,
        "kernel_runtime_dispatches": kernel_runtime_dispatches,
        "predicted_tflops_s": pred_tflops_s,
        "measured_tflops_s": meas_tflops_s,
        "predicted_tb_s": pred_tb_s,
        "measured_tb_s": meas_tb_s,
        "throughput_ops_counter": measured_throughput_label,
    }


def _run_prediction(predict):
    """Run the perf model through ``predict()`` and print what it returned."""
    predicted_flops, predicted_bytes, predicted_precision = predict()
    print(f'       Predicted FLOPs: {format_number(predicted_flops, "ops")}')
    print(f'       Predicted Bytes: {format_number(predicted_bytes, "bytes")}')
    print(f'       Compute precision: {predicted_precision or "unknown"}')
    return predicted_flops, predicted_bytes, predicted_precision


def _unmeasured_row(op_name, run_status, extra_fields, predicted=(None, None, "---")):
    """Result row for an op that has no hardware measurement."""
    predicted_flops, predicted_bytes, predicted_precision = predicted
    return {
        "op": op_name,
        **extra_fields,
        "predicted_flops": predicted_flops,
        "predicted_bytes": predicted_bytes,
        "predicted_precision": predicted_precision,
        "rocprof_f4_ops": 0,
        "rocprof_f8_ops": 0,
        "rocprof_total_16": 0,
        "rocprof_total_32": 0,
        "rocprof_read_bytes": 0,
        "rocprof_write_bytes": 0,
        "kernel_name": "",
        "flops_ratio": None,
        "bytes_ratio": None,
        "precision": predicted_precision,
        "run_status": run_status,
    }


def _fail_row(op_name, error, extra_fields):
    """Result row for an op whose validation raised ``error``."""
    return _unmeasured_row(op_name, f"FAIL: {str(error)[:60]}", extra_fields)


def _report_model_only(op_name, arch, output_dir, dims_str, predict, extra_fields):
    """Report an op that needs model-level context from its prediction alone."""
    print("Running TraceLens perf model (perf-model-only mode) ...")
    predicted_flops, predicted_bytes, predicted_precision = _run_prediction(predict)
    lines = [
        "======================================================================",
        "PERF MODEL VALIDATION REPORT (perf-model-only)",
        "======================================================================",
        f"Op:   {op_name}",
        f"Arch: {arch}",
        dims_str,
        "",
        "--- Perf Model Predictions ---",
        f'  FLOPs:               {format_number(predicted_flops, "ops")}',
        f'  Bytes (roofline min): {format_number(predicted_bytes, "bytes")}',
        f'  Compute precision:   {predicted_precision or "unknown"}',
        "",
        "Note: This op requires model-level context (ForwardContext, KV cache)",
        "and cannot be tested standalone with rocprofv3. Only the perf model",
        "predictions are shown.",
        "======================================================================",
    ]
    report_text = "\n".join(lines)
    print(report_text)
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    predicted = (predicted_flops, predicted_bytes, predicted_precision or "N/A")
    return _unmeasured_row(op_name, "MODEL_ONLY", extra_fields, predicted)


def _collect_counters(args, runner_argv, kernel_filter, output_dir):
    """Run the rocprofv3 counter passes and return ``(ops_counters, mem_counters)``.

    The MFMA ops pass must succeed. A derived-ops or memory pass that fails is
    reported and leaves its counters out.
    """

    def collect(key):
        csv_path = run_rocprofv3(
            args.rocprofv3_path,
            ["--pmc", *ARCH_COUNTER_CONFIGS[args.arch][key]],
            "counter_collection.csv",
            runner_argv,
            key,
            output_dir,
            args.timeout,
        )
        return parse_counter_csv(csv_path, kernel_filter)

    ops_counters = collect("ops_counters")
    mem_counters = {}
    passes = [("mem_read_counters", mem_counters), ("mem_write_counters", mem_counters)]
    if not args.skip_derived:
        passes = [
            ("ops_derived_16", ops_counters),
            ("ops_derived_32", ops_counters),
            *passes,
        ]
    for key, counters in passes:
        try:
            counters.update(collect(key))
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  WARNING: {key} collection failed: {e}")
    return ops_counters, mem_counters


def run_op_validation(
    op_name, args, output_dir, title, dims_str, predict, build_argv, extra_fields
):
    """Validate one op against rocprofv3 counters; return its result row.

    ``op_name`` is the OP_REGISTRY key. ``predict()`` returns the perf model's
    ``(flops, bytes, precision)`` and ``build_argv()`` returns the runner argv;
    ops registered ``perf_model_only`` skip the latter.
    ``extra_fields`` are added to the row. A rocprofv3 failure raises.
    """
    print("\n" + "#" * 70)
    print(f"# Validating: {title}")
    print(f"# Dims: {dims_str}")
    print("#" * 70)
    if OP_REGISTRY[op_name].get("perf_model_only"):
        return _report_model_only(
            op_name, args.arch, output_dir, dims_str, predict, extra_fields
        )
    print("Running TraceLens perf model ...")
    predicted_flops, predicted_bytes, predicted_precision = _run_prediction(predict)
    print("Building replay runner command ...")
    runner_argv = build_argv()
    print(f'       Runner cmd: {" ".join(runner_argv)}')
    kernel_filter = args.kernel_filter
    if kernel_filter:
        print(f"Using manual kernel filter: {kernel_filter}")
    print("Tracing the measured call via rocprofv3 --kernel-trace ...")
    trace_csv_path, kernel_names = discover_measured_kernels(
        args.rocprofv3_path,
        runner_argv,
        output_dir,
        timeout=args.timeout,
        kernel_filter=kernel_filter,
    )
    kernel_label = "; ".join(kernel_names)
    ops_counters, mem_counters = _collect_counters(
        args, runner_argv, kernel_filter, output_dir
    )
    runtime_ns, runtime_mean_ns, runtime_dispatches = parse_kernel_runtime_ns(
        trace_csv_path, kernel_filter
    )
    report_data = generate_report(
        op_name,
        args.arch,
        dims_str,
        predicted_flops,
        predicted_bytes,
        predicted_precision,
        ops_counters,
        mem_counters,
        output_dir,
        kernel_filter=kernel_label,
        kernel_runtime_ns=runtime_ns,
        kernel_runtime_mean_ns=runtime_mean_ns,
        kernel_runtime_dispatches=runtime_dispatches,
    )
    return {
        "op": op_name,
        **extra_fields,
        "predicted_flops": predicted_flops,
        "predicted_bytes": predicted_bytes,
        "predicted_precision": predicted_precision or "N/A",
        **report_data,
        "kernel_name": kernel_label,
        "run_status": "OK",
    }


def run_single_op_validation(op_name, args, output_dir):
    """Run full validation for a single registered op. Returns result dict."""
    reg = OP_REGISTRY[op_name]
    effective = argparse.Namespace(**vars(args))
    effective.op = op_name
    for key, default_val in reg["defaults"].items():
        if getattr(effective, key, None) is None:
            setattr(effective, key, default_val)
    parts = []
    for key in (
        "M",
        "N",
        "K",
        "E",
        "topk",
        "group_size",
        "seq_len",
        "num_heads_q",
        "num_heads_kv",
        "head_dim",
    ):
        val = getattr(effective, key, None)
        if val is not None and key in reg["required_args"]:
            parts.append(f"{key}={val}")
    ann_val = getattr(effective, "annotation", None)
    if ann_val:
        parts.append(f'annotation={ann_val[:60]}{"..." if len(ann_val) > 60 else ""}')

    _populate_varlen_attention_args(op_name, effective)
    return run_op_validation(
        op_name,
        args,
        output_dir,
        title=f'{op_name} ({reg["description"]})',
        dims_str=", ".join(parts),
        predict=lambda: reg["model_fn"](effective),
        build_argv=lambda: _build_runner_argv(op_name, effective),
        extra_fields={},
    )


def load_cases(cases_path):
    """Load cases written by ``bringup_gates cases``; keep those with a harness."""
    entries = []
    for case in json.loads(Path(cases_path).read_text(encoding="utf-8")):
        registry_key = TRACE_NAME_TO_OP.get(case["trace_name"])
        if registry_key is None:
            print(f"  Skipped {case['trace_name']} (no registry entry)")
            continue
        if OP_REGISTRY[registry_key]["category"] not in ("attention", "vllm_attention"):
            case["attn_params"] = None
        entries.append({**case, "registry_key": registry_key})
    return entries


def _case_fields(entry):
    """Result-row fields that identify a ``--cases`` entry."""
    return {
        "trace_name": entry["trace_name"],
        "csv_reported_flops": entry["csv_flops"],
        "csv_reported_bytes": entry["csv_bytes"],
    }


def run_csv_op_validation(entry, args, output_dir):
    """Run validation for a single traced case from ``--cases``."""
    registry_key = entry["registry_key"]
    return run_op_validation(
        registry_key,
        args,
        output_dir,
        title=f'{entry["trace_name"]} -> {registry_key}',
        dims_str=entry["dims_str"][:80] if entry["dims_str"] else "",
        predict=lambda: run_perf_model_from_event(
            entry["trace_name"],
            entry["input_dims"],
            entry["input_types"],
            entry["input_strides"],
            entry["concrete_inputs"],
            entry.get("perf_params") or entry["attn_params"],
        ),
        build_argv=lambda: _build_runner_argv_from_csv_entry(
            registry_key,
            entry["input_dims"],
            entry["input_types"],
            entry["concrete_inputs"],
            entry["attn_params"],
            perf_params=entry.get("perf_params"),
        ),
        extra_fields=_case_fields(entry),
    )


def write_results_csv(results, output_path):
    """Write the enriched results table as a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {}
            for col in CSV_COLUMNS:
                val = r.get(col, "")
                if val is None:
                    val = ""
                elif isinstance(val, float):
                    if col in ("flops_ratio", "bytes_ratio"):
                        val = f"{val:.4f}"
                    elif col in (
                        "predicted_tflops_s",
                        "measured_tflops_s",
                        "predicted_tb_s",
                        "measured_tb_s",
                    ):
                        val = f"{val:.4f}"
                    elif col in ("kernel_runtime_us", "kernel_runtime_mean_us"):
                        val = f"{val:.3f}"
                    else:
                        val = f"{val:.0f}"
                row[col] = val
            writer.writerow(row)
    print(f"\nCSV report saved to: {output_path}")
    return None


def print_batch_summary(results):
    """Print an enriched summary table for batch validation runs."""
    print(
        "\n===================================================================================================================================================================================="
    )
    print("VALIDATION REPORT TABLE")
    print(
        "===================================================================================================================================================================================="
    )
    hdr = (
        f'  {"Op":<32s} {"Class":<32s} {"Source":<42s} {"Prec":>5s} {"Pred FLOPs":>12s} '
        f'{"Pred Bytes":>12s} {"F4 Ops":>12s} {"F8 Ops":>12s} {"16-bit":>12s} {"32-bit":>12s} '
        f'{"Read":>12s} {"Write":>12s} {"FRatio":>8s} {"BRatio":>8s} {"Run":>10s} {"Match":>14s}'
    )
    print(hdr)
    sep = (
        f'  {"--------------------------------"} {"--------------------------------"} '
        f'{"------------------------------------------"} {"-----"} {"------------"} {"------------"} '
        f'{"------------"} {"------------"} {"------------"} {"------------"} {"------------"} '
        f'{"------------"} {"--------"} {"--------"} {"----------"} {"--------------"}'
    )
    print(sep)
    for r in results:
        fr_s = f'{r["flops_ratio"]:.4f}' if r.get("flops_ratio") else "---"
        br_s = f'{r["bytes_ratio"]:.4f}' if r.get("bytes_ratio") else "---"
        pf = (
            format_number(r.get("predicted_flops"), "ops")
            if r.get("predicted_flops")
            else "---"
        )
        pb = (
            format_number(r.get("predicted_bytes"), "bytes")
            if r.get("predicted_bytes")
            else "---"
        )
        f4 = (
            format_number(r.get("rocprof_f4_ops", 0), "ops")
            if r.get("rocprof_f4_ops")
            else "---"
        )
        f8 = (
            format_number(r.get("rocprof_f8_ops", 0), "ops")
            if r.get("rocprof_f8_ops")
            else "---"
        )
        t16 = (
            format_number(r.get("rocprof_total_16", 0), "ops")
            if r.get("rocprof_total_16")
            else "---"
        )
        t32 = (
            format_number(r.get("rocprof_total_32", 0), "ops")
            if r.get("rocprof_total_32")
            else "---"
        )
        read = (
            format_number(r.get("rocprof_read_bytes", 0), "bytes")
            if r.get("rocprof_read_bytes")
            else "---"
        )
        write = (
            format_number(r.get("rocprof_write_bytes", 0), "bytes")
            if r.get("rocprof_write_bytes")
            else "---"
        )
        prec = str(r.get("predicted_precision", "---"))[:5]
        src = f'{r.get("source_file", "?")}:{r.get("source_line", "?")}'
        cls_name = str(r.get("perf_model_class", "?"))[:32]
        status = str(r.get("run_status", "---"))[:10]
        match = str(r.get("match_status") or "---")[:14]
        op_name = r["op"][:32]
        line = (
            f"  {op_name:<32s} {cls_name:<32s} {src:<42s} {prec:>5s} {pf:>12s} {pb:>12s} "
            f"{f4:>12s} {f8:>12s} {t16:>12s} {t32:>12s} {read:>12s} {write:>12s} "
            f"{fr_s:>8s} {br_s:>8s} {status:>10s} {match:>14s}"
        )
        print(line)
    print(
        "===================================================================================================================================================================================="
    )
    print("\nDISCREPANCY COMMENTS:")
    print(
        "----------------------------------------------------------------------------------------------------"
    )
    for r in results:
        comment = r.get("comment", "")
        if comment and comment != "OK":
            print(f'  {r["op"]:<35s}  {comment}')
    print(
        "----------------------------------------------------------------------------------------------------"
    )
    ok_count = sum(1 for r in results if r["run_status"] == "OK")
    model_only_count = sum(1 for r in results if r["run_status"] == "MODEL_ONLY")
    fail_count = len(results) - ok_count - model_only_count
    parts = [f"{ok_count} passed"]
    if model_only_count:
        parts.append(f"{model_only_count} model-only")
    if fail_count:
        parts.append(f"{fail_count} failed")
    print(f'\n  {", ".join(parts)} out of {len(results)} ops')
    matches = Counter(r.get("match_status") for r in results if r.get("match_status"))
    if matches:
        print("  match: " + ", ".join(f"{n} {k}" for k, n in sorted(matches.items())))
    return None


def discover_missing_coverage():
    """Scan TraceLens perf model mappings for aiter:: and vllm:: ops and report coverage."""
    try:
        from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import (
            get_pseudo_op_mappings,
        )
        from TraceLens.PerfModel.torch_op_mapping import op_to_perf_model_class_map
    except ImportError as e:
        print(f"  WARNING: Could not import perf model mappings: {e}")
        return None
    all_ops = {}
    for name, cls in get_pseudo_op_mappings().items():
        if name.startswith("aiter::") or name.startswith("vllm::"):
            all_ops[name] = cls.__name__
    for name, cls in op_to_perf_model_class_map.items():
        if name.startswith("aiter::") or name.startswith("vllm::"):
            all_ops[name] = cls.__name__
    SKIP_PATTERNS = [
        "backward",
        "bwd",
        "reduce_scatter",
        "all_gather",
        "fused_allreduce",
    ]
    registry_trace_names = set(TRACE_NAME_TO_OP) | {
        f"aiter::{key}" for key in OP_REGISTRY
    }
    print("\n======================================================================")
    print("AITER:: & VLLM:: OP COVERAGE REPORT")
    print("======================================================================")
    covered = []
    missing = []
    skipped = []
    for trace_name in sorted(all_ops):
        cls_name = all_ops[trace_name]
        if any(p in trace_name.lower() for p in SKIP_PATTERNS):
            skipped.append((trace_name, cls_name, "skipped (collective/backward)"))
            continue
        short_name = trace_name.replace("aiter::", "").replace("vllm::", "")
        if trace_name in registry_trace_names or short_name in OP_REGISTRY:
            covered.append((trace_name, cls_name))
        else:
            missing.append((trace_name, cls_name))
    print(f"\nCovered ({len(covered)}):")
    for name, cls in covered:
        print(f"  [OK]   {name:50s}  -> {cls}")
    if missing:
        print(f"\nMissing ({len(missing)}):")
        for name, cls in missing:
            print(f"  [--]   {name:50s}  -> {cls}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for name, cls, reason in skipped:
            print(f"  [skip] {name:50s}  ({reason})")
    print("======================================================================")
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Validate TraceLens perf model against rocprofv3 hardware counters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            "            Available ops:  %(ops)s\n            Categories:     %(cats)s\n        "
            % {
                "ops": ", ".join(sorted(OP_REGISTRY.keys())),
                "cats": ", ".join(CATEGORIES),
            }
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--op", choices=list(OP_REGISTRY.keys()), help="Single op to validate."
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all registered ops with default dimensions.",
    )
    group.add_argument(
        "--category", choices=CATEGORIES, help="Run all ops in a category."
    )
    group.add_argument(
        "--discover",
        action="store_true",
        help="List all aiter:: ops and their validation coverage, then exit.",
    )
    group.add_argument(
        "--cases",
        metavar="JSON",
        help="Validate the traced shapes in this cases file (from bringup_gates cases).",
    )
    parser.add_argument("--M", type=int, default=None, help="M dimension (tokens)")
    parser.add_argument(
        "--N", type=int, default=None, help="N dimension (inter_dim / columns)"
    )
    parser.add_argument(
        "--K", type=int, default=None, help="K dimension (hidden_dim / rows)"
    )
    parser.add_argument(
        "--E", type=int, default=None, help="Number of experts (MoE ops)"
    )
    parser.add_argument(
        "--topk", type=int, default=None, help="Top-k experts per token (MoE ops)"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="Quantization group size (vllm quant/norm ops, default 128)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=None, help="Sequence length (attention ops)"
    )
    parser.add_argument(
        "--num_heads_q", type=int, default=None, help="Query heads (attention ops)"
    )
    parser.add_argument(
        "--num_heads_kv", type=int, default=None, help="KV heads (attention ops)"
    )
    parser.add_argument(
        "--head_dim", type=int, default=None, help="Head dimension (attention ops)"
    )
    parser.add_argument(
        "--annotation",
        default=None,
        help="Detailed step annotation passed through to ops that accept it "
        "(e.g. unified_attention), in any format TraceLens parses: vLLM "
        "'execute_<i>_context_<N>(sq<S>sk<K>sqsq<Q>sqsk<P>)_generation_<M>(...)', "
        "SGLang 'step[... c_sq=.. g_sq=.. ...]' or ATOM detailed.",
    )
    parser.add_argument(
        "--varlen_num_seqs",
        type=int,
        default=None,
        dest="varlen_num_seqs",
        help="Number of sequences packed into a varlen attention call (default 4).",
    )
    parser.add_argument(
        "--varlen_seed",
        type=int,
        default=None,
        dest="varlen_seed",
        help="RNG seed for the random varlen partition (default 42).",
    )
    parser.add_argument(
        "--varlen_scenario",
        default=None,
        dest="varlen_scenario",
        choices=["random", "mixed_prefill_decode"],
        help="'random' (default): self-attention with random partition of seq_len "
        "tokens. 'mixed_prefill_decode': 1 prefill seq with Q=K=seq_len + "
        "(varlen_num_seqs-1) decode seqs with Q=1, K=seq_len.",
    )
    DTYPE_CHOICES = [
        "bf16",
        "fp16",
        "fp32",
        "fp8",
        "fp8_e4m3fn",
        "fp8_e4m3fnuz",
        "fp8_e5m2",
        "fp4",
        "fp4x2",
        "i8",
        "u8",
    ]
    parser.add_argument(
        "--in_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Activation/input dtype (default per-op; see harnesses).",
    )
    parser.add_argument(
        "--w_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Weight dtype (default per-op).",
    )
    parser.add_argument(
        "--out_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Output dtype (default per-op).",
    )
    parser.add_argument(
        "--kv_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Paged-attention KV-cache dtype (default per-op).",
    )
    parser.add_argument(
        "--bias_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="GEMM bias dtype (used by vllm_unquantized_gemm bias path).",
    )
    parser.add_argument(
        "--scale_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Block-scale dtype (passed to the GPU test harness).",
    )
    parser.add_argument(
        "--quant_dtype",
        default=None,
        choices=DTYPE_CHOICES,
        help="Quantization storage dtype.",
    )
    parser.add_argument(
        "--quant_type",
        default=None,
        choices=[
            "no",
            "per_tensor",
            "per_token",
            "per_1x32",
            "per_1x128",
            "per_128x128",
            "per_256x128",
            "per_1024x128",
        ],
        help="aiter QuantType selector for ck_moe_stage{1,2}.",
    )
    parser.add_argument(
        "--activation",
        default=None,
        choices=["silu", "gelu", "swiglu", "no"],
        help="Activation selector (silu/gelu/swiglu).",
    )
    parser.add_argument(
        "--block_n",
        type=int,
        default=None,
        help="Per-1x128 / per-128x128 N tile size (default 128).",
    )
    parser.add_argument(
        "--block_k",
        type=int,
        default=None,
        help="Per-1x128 / per-128x128 K tile size (default 128).",
    )
    parser.add_argument(
        "--block_m", type=int, default=None, help="MoE M-block size (default 32)."
    )
    parser.add_argument(
        "--split_k",
        type=int,
        default=None,
        help="CK-Tile / CK split-K factor (default 1).",
    )
    parser.add_argument(
        "--arch",
        default="gfx942",
        choices=list(ARCH_COUNTER_CONFIGS.keys()),
        help="GPU architecture (default: gfx942).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for harness, rocprofv3 output, and report.",
    )
    parser.add_argument(
        "--rocprofv3-path", default="rocprofv3", help="Path to rocprofv3 binary."
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip derived counters (TOTAL_16_OPS, TOTAL_32_OPS).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each rocprofv3 run.",
    )
    parser.add_argument(
        "--kernel-filter",
        default=None,
        help="Manual kernel name filter (overrides auto-discovery).",
    )
    parser.add_argument(
        "--only-ops",
        default=None,
        help="Comma-separated registry keys or trace names; with --cases, "
        "restrict the run to those ops (useful for targeted re-runs).",
    )
    args = parser.parse_args()
    if os.sep in args.rocprofv3_path:
        args.rocprofv3_path = os.path.abspath(args.rocprofv3_path)

    if args.discover:
        discover_missing_coverage()
        return None

    if args.cases:
        if not args.output_dir:
            parser.error("--output-dir is required for validation runs.")
        base_output_dir = os.path.abspath(args.output_dir)
        os.makedirs(base_output_dir, exist_ok=True)
        entries = load_cases(args.cases)
        if args.only_ops:
            wanted = {o.strip() for o in args.only_ops.split(",") if o.strip()}
            entries = [
                e
                for e in entries
                if e["registry_key"] in wanted or e["trace_name"] in wanted
            ]
            print(f"--only-ops filter kept {len(entries)} entries: {sorted(wanted)}")
        if not entries:
            print("No cases to validate.")
            return None
        results = []
        for idx, entry in enumerate(entries):
            safe_name = re.sub("[^a-zA-Z0-9_]", "_", entry["trace_name"])
            op_output_dir = os.path.join(base_output_dir, f"{idx:03d}_{safe_name}")
            os.makedirs(op_output_dir, exist_ok=True)
            try:
                result = run_csv_op_validation(entry, args, op_output_dir)
                results.append(result)
            except Exception as e:
                print(f'\n  FAILED: {entry["trace_name"]}: {e}')
                results.append(_fail_row(entry["registry_key"], e, _case_fields(entry)))
        _enrich_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"validation_report_{timestamp}.csv"
        csv_path = os.path.join(base_output_dir, csv_filename)
        write_results_csv(results, csv_path)
        print_batch_summary(results)
        return int(
            any(str(r.get("run_status", "")).startswith("FAIL") for r in results)
        )

    if args.all:
        ops_to_run = list(OP_REGISTRY.keys())
    elif args.category:
        ops_to_run = [
            k for k, v in OP_REGISTRY.items() if v["category"] == args.category
        ]
    elif args.op:
        ops_to_run = [args.op]
    else:
        parser.error(
            "One of --op, --all, --category, --discover, or --cases is required."
        )

    if not args.output_dir:
        parser.error("--output-dir is required for validation runs.")
    base_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    is_batch = len(ops_to_run) > 1
    results = []
    for op_name in ops_to_run:
        if is_batch:
            op_output_dir = os.path.join(base_output_dir, op_name)
        else:
            op_output_dir = base_output_dir
        os.makedirs(op_output_dir, exist_ok=True)
        try:
            result = run_single_op_validation(op_name, args, op_output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n  FAILED: {op_name}: {e}")
            results.append(_fail_row(op_name, e, {}))
    _enrich_results(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"validation_report_{timestamp}.csv"
    csv_path = os.path.join(base_output_dir, csv_filename)
    write_results_csv(results, csv_path)
    print_batch_summary(results)
    return None


def _build_runner_argv(op_name, effective_args, num_warmup=3):
    """Build the argv list that rocprofv3 wraps for op ``op_name``."""
    cmd = [
        sys.executable,
        str(REPLAY_RUNNER),
        "--op",
        op_name,
        "--num-warmup",
        str(num_warmup),
    ]
    for k in INT_FLAGS:
        v = getattr(effective_args, k, None)
        if v is not None:
            cmd += [f"--{k}", str(v)]
    for k in STR_FLAGS:
        v = getattr(effective_args, k, None)
        if v is not None and v != "":
            cmd += [f"--{k}", str(v)]
    varlen_seed = getattr(effective_args, "varlen_seed", None)
    if varlen_seed is not None:
        cmd += ["--varlen-seed", str(varlen_seed)]
    varlen_num_seqs = getattr(effective_args, "varlen_num_seqs", None)
    if varlen_num_seqs is not None:
        cmd += ["--varlen-num-seqs", str(varlen_num_seqs)]
    varlen_scenario = getattr(effective_args, "varlen_scenario", None)
    if varlen_scenario:
        cmd += ["--varlen-scenario", str(varlen_scenario)]
    ann = getattr(effective_args, "annotation", None)
    if ann:
        cmd += ["--annotation", ann]
    return cmd


def _build_csv_runner_argv(
    registry_key,
    effective_args,
    num_warmup=3,
    input_dims=None,
    input_types=None,
    concrete_inputs=None,
):
    """Build a runner argv for the generic CSV path.

    ``registry_key`` is the OP_REGISTRY key (also used by
    ``test_generic_simple_op`` to look up the call dispatch); ``input_dims``
    and ``input_types`` are the raw lists from ``unified_perf_summary.csv``.
    ``concrete_inputs`` carries the traced non-tensor arguments (expert counts,
    block sizes, flags) so dispatchers can reproduce the original call.
    """
    cmd = [
        sys.executable,
        str(REPLAY_RUNNER),
        "--op",
        "__generic__",
        "--registry-key",
        registry_key,
        "--num-warmup",
        str(num_warmup),
    ]
    if input_dims is not None:
        cmd += ["--input-dims-json", json.dumps(tuple_to_list(input_dims))]
    if input_types is not None:
        cmd += [
            "--input-types-json",
            json.dumps(list(input_types) if input_types else []),
        ]
    if concrete_inputs is not None:
        cmd += ["--concrete-inputs-json", json.dumps([str(c) for c in concrete_inputs])]
    return cmd


def _populate_varlen_attention_args(op_name, effective_args):
    """Pre-compute c_sq/c_sqsk on the parent side for varlen attention ops.

    The test runs in a rocprofv3 subprocess, so the parent recomputes the
    per-batch aggregates the perf model needs from the same deterministic
    seed + scenario used by ``replay/attention.make_varlen_seqlens``. Also
    pre-computes total Q / KV token counts so the event builders can size
    Input Dims correctly for non-self-attention scenarios (mixed
    prefill+decode).
    """
    if op_name not in ("mha_varlen_fwd", "fmha_v3_varlen_fwd"):
        return None
    if getattr(effective_args, "_varlen_c_sq", None) is not None:
        return None
    num_seqs = getattr(effective_args, "varlen_num_seqs", None) or 4
    seed = getattr(effective_args, "varlen_seed", None) or 42
    scenario = getattr(effective_args, "varlen_scenario", None) or "random"
    sq, cu_q, sk, cu_k = make_varlen_seqlens(
        effective_args.seq_len, num_seqs=num_seqs, seed=seed, scenario=scenario
    )
    c_sq, c_sqsk = compute_varlen_annotation_stats(sq, sk)
    effective_args._varlen_c_sq = c_sq
    effective_args._varlen_c_sqsq = c_sqsk
    effective_args._varlen_max_seqlen = max(sq + sk)
    effective_args._varlen_total_q = cu_q[-1]
    effective_args._varlen_total_kv = cu_k[-1]
    return None


def _extract_dims_for_existing_harness(
    registry_key,
    input_dims,
    input_types,
    concrete_inputs,
    attn_params=None,
    perf_params=None,
):
    """Extract M/N/K/etc. from Input Dims for ops that use existing harness generators."""
    ns = argparse.Namespace()
    reg = OP_REGISTRY[registry_key]
    cat = reg["category"]
    pp = perf_params or {}
    if cat == "dsv3_mla_decode":
        # pseudo_mla_decode_fwd carries a whole scheduler batch: c_* describe the
        # context (prefill) requests and g_* the generation requests. Both halves
        # run in one kernel launch, so the harness has to rebuild both.
        nhead = int(pp.get("H_Q") or 16)
        qk = int(pp.get("d_h_qk") or 576)
        n_gen = int(pp.get("g_sq") or pp.get("N_Q") or 64)
        g_sk = int(pp.get("g_sk") or (n_gen * 8677))
        # c_sqsq / c_sq is the (uniform) query length of a context request. The
        # c_* terms describe the scheduler batch, which is not always the same
        # thing as this call: some backends route context attention to a
        # separate flash-attention op and only send generation tokens here. The
        # query tensor settles it -- replay the context half only when N_Q
        # actually accounts for those tokens.
        c_sq = int(pp.get("c_sq") or 0)
        c_sqsq = int(pp.get("c_sqsq") or 0)
        n_q = int(pp.get("N_Q") or n_gen)
        ctx_qlen = round(c_sqsq / c_sq) if c_sq else 0
        ctx_in_this_call = c_sq and abs(n_q - (c_sq + n_gen)) <= 1
        ns.n_ctx = round(c_sq / ctx_qlen) if (ctx_qlen and ctx_in_this_call) else 0
        ns.ctx_qlen = ctx_qlen if ctx_in_this_call else 0
        ns.E = n_gen
        ns.seq_len = max(1, round(g_sk / n_gen)) if n_gen else 8677
        ns.num_heads_q = nhead
        ns.head_dim = qk
        # The KV cache tensor dtype picks the a8w8 or a16w16 kernel; the logical
        # dtype in perf_params says bf16 even where the cache is quantised.
        kv_type = str(input_types[1]) if input_types and len(input_types) > 1 else ""
        ns.kv_dtype = "fp8" if "Float8" in kv_type else "bf16"
        return ns
    if cat == "dsv3_moe_flydsl":
        ns.M = int(pp.get("num_tokens") or 64)
        ns.K = int(pp.get("hidden_dim") or 7168)
        ns.N = int(pp.get("inter_dim") or 256)
        ns.E = int(pp.get("num_experts") or 257)
        ns.topk = int(pp.get("topk") or 9)
        ns.block_m = 32
        return ns
    if cat in ("gemm", "vllm_gemm"):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        ns.K = (
            input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 8192
        )
        if len(input_dims) > 1 and input_dims[1]:
            ns.N = input_dims[1][0]
        else:
            ns.N = 4096
        return ns
    if cat == "moe":
        # perf_params carries the logical MoE dims directly. The raw operand
        # shapes are ambiguous (operand 0 may be the routing table rather than
        # the activation, which yields K=topk), so prefer perf_params.
        if pp.get("num_tokens") and pp.get("hidden_dim"):
            ns.M = int(pp["num_tokens"])
            ns.K = int(pp["hidden_dim"])
            ns.N = int(pp.get("inter_dim") or 256)
            ns.E = int(pp.get("num_experts") or 8)
            ns.topk = int(pp.get("topk") or 2)
            return ns
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 512
        ns.K = (
            input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 7168
        )
        if len(input_dims) > 1 and input_dims[1] and len(input_dims[1]) >= 3:
            ns.E = input_dims[1][0]
            ns.N = input_dims[1][2]
        else:
            ns.E = 8
            ns.N = 2048
        if concrete_inputs:
            for ci in concrete_inputs:
                if isinstance(ci, (int, float)) and ci not in (0, 1) and ci < 100:
                    ns.topk = int(ci)
                    break
            else:
                ns.topk = 2
            return ns
        ns.topk = 2
        return ns
    if cat in ("activation",):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        if len(input_dims) > 0 and len(input_dims[0]) > 1:
            ns.N = input_dims[0][1]
            return ns
        ns.N = 7168
        return ns
    if cat in ("norm", "vllm_norm"):
        ns.M = input_dims[0][0] if len(input_dims) > 0 and input_dims[0] else 2048
        ns.N = (
            input_dims[0][1] if len(input_dims) > 0 and len(input_dims[0]) > 1 else 7168
        )
        ns.group_size = 128
        return ns
    if cat in ("quant", "vllm_quant"):
        if len(input_dims) > 0 and input_dims[0]:
            if len(input_dims[0]) == 3:
                ns.M = input_dims[0][0] * input_dims[0][1]
                ns.N = input_dims[0][2]
            else:
                ns.M = input_dims[0][0]
                ns.N = input_dims[0][1] if len(input_dims[0]) > 1 else 7168
        else:
            ns.M, ns.N = (2048, 7168)
        ns.group_size = 128
        return ns
    if cat in ("attention", "vllm_attention"):
        # The perf model's own parsed params come first, so the harness runs
        # the same sq/sk the model predicted for.
        params = pp or attn_params
        if params:
            ns.seq_len = int(params.get("N_Q", 2048))
            ns.num_heads_q = int(params.get("H_Q", 32))
            ns.num_heads_kv = int(params.get("H_KV", 8))
            ns.head_dim = int(params.get("d_h_qk", 128))
            c_sq = params.get("c_sq")
            c_sqsq = params.get("c_sqsq")
            if c_sq is not None and str(c_sq) not in ("nan", ""):
                ns._varlen_c_sq = int(float(c_sq))
            if c_sqsq is not None and str(c_sqsq) not in ("nan", ""):
                ns._varlen_c_sqsq = int(float(c_sqsq))
            ns.annotation = chunk_stats_annotation(params)
            return ns
        if len(input_dims) > 0 and input_dims[0] and len(input_dims[0]) >= 3:
            dims = input_dims[0]
            if len(dims) == 4:
                ns.seq_len = dims[1]
                ns.num_heads_q = dims[2]
                ns.head_dim = dims[3]
            elif len(dims) == 3:
                ns.seq_len = dims[0]
                ns.num_heads_q = dims[1]
                ns.head_dim = dims[2]
            else:
                ns.seq_len, ns.num_heads_q, ns.head_dim = (2048, 32, 128)
        else:
            ns.seq_len, ns.num_heads_q, ns.head_dim = (2048, 32, 128)
        ns.num_heads_kv = 8
        if len(input_dims) > 1 and input_dims[1] and len(input_dims[1]) >= 3:
            kv_dims = input_dims[1]
            if len(kv_dims) == 4:
                ns.num_heads_kv = kv_dims[2]
                return ns
            if len(kv_dims) == 3:
                ns.num_heads_kv = kv_dims[1]
        return ns
    for key, default_val in reg.get("defaults", {}).items():
        setattr(ns, key, default_val)
    return ns


def _build_runner_argv_from_csv_entry(
    registry_key,
    input_dims,
    input_types,
    concrete_inputs,
    attn_params=None,
    num_warmup=3,
    perf_params=None,
):
    """Decide which runner invocation to use for a CSV-driven validation entry.

    If the op was registered with ``csv_harness=True``, dimensions
    are extracted from ``input_dims`` / ``input_types`` (and the optional
    ``attn_params``) and the standard per-op test_<op> is invoked. Otherwise
    the generic ``__generic__`` test is dispatched with the raw CSV shapes.
    """
    if OP_REGISTRY[registry_key]["csv_harness"]:
        ns = _extract_dims_for_existing_harness(
            registry_key,
            input_dims,
            input_types,
            concrete_inputs,
            attn_params,
            perf_params,
        )
        return _build_runner_argv(registry_key, ns, num_warmup=num_warmup)
    return _build_csv_runner_argv(
        registry_key,
        argparse.Namespace(),
        num_warmup=num_warmup,
        input_dims=input_dims,
        input_types=input_types,
        concrete_inputs=concrete_inputs,
    )


def _generate_comment(row):
    """Auto-generate a discrepancy comment from rule-based analysis."""
    if row.get("run_status", "").startswith("FAIL"):
        return f'Validation failed: {row["run_status"]}'
    if row.get("run_status") == "MODEL_ONLY":
        return "Perf-model-only; no GPU measurement available"
    parts = []
    fr = row.get("flops_ratio")
    br = row.get("bytes_ratio")
    fr_low, fr_high = RATIO_MATCH["flops_ratio"]
    br_low, br_high = RATIO_MATCH["bytes_ratio"]
    if fr is not None:
        if fr_low <= fr <= fr_high:
            parts.append("FLOPs: exact match")
        elif 0.8 <= fr < fr_low:
            parts.append(
                f"FLOPs {fr:.2f}x: minor under-count, likely rounding or tile residuals"
            )
        elif fr < 0.5:
            parts.append(
                f"FLOPs {fr:.2f}x: kernel may use different MFMA dtype than predicted"
            )
        elif fr < 0.8:
            parts.append(
                f"FLOPs {fr:.2f}x: possible mixed-precision or partial tile usage"
            )
        elif fr_high < fr <= 1.5:
            parts.append(
                f"FLOPs {fr:.2f}x: minor over-count from tile padding or epilogue"
            )
        elif fr > 2:
            parts.append(
                f"FLOPs {fr:.1f}x: likely tile padding, SwiGLU fusion, or extra VALU ops"
            )
        else:
            parts.append(f"FLOPs {fr:.2f}x: moderate over-count, check tile dimensions")
    if br is not None:
        if br_low <= br <= br_high:
            parts.append("Bytes: good match")
        elif br < br_low:
            parts.append(f"Bytes {br:.2f}x: model over-predicts bytes")
        elif br_high < br <= 2:
            parts.append(f"Bytes {br:.2f}x: L2 replay or scale tensor overhead")
        elif br > 2:
            parts.append(
                f"Bytes {br:.1f}x: significant extra traffic (L2 replays, scales, workspace)"
            )
    prec = row.get("precision", "")
    if prec == "MISMATCH":
        parts.append(
            f"Precision mismatch: {row['precision_reason']}"
            if row.get("precision_reason")
            else "Precision mismatch: dominant HW counter differs from predicted dtype"
        )
    elif prec == "NO DATA":
        parts.append("No ops counters collected for precision check")
    if parts:
        return "hint: " + "; ".join(parts)
    return "OK"


def _enrich_results(results):
    """Add perf-model source metadata and auto-generated comments to results."""
    for r in results:
        cls_name, src_file, src_line = model_source(r["op"])
        r["perf_model_class"] = cls_name
        r["source_file"] = src_file
        r["source_line"] = src_line
        r["comment"] = _generate_comment(r)
        for ratio_col, band_col in BAND_COLUMNS.items():
            r[band_col] = ratio_band(ratio_col, r.get(ratio_col))
        r["match_status"] = match_status(
            r.get("run_status"), [r[band] for band in BAND_COLUMNS.values()]
        )
    return results


if __name__ == "__main__":
    sys.exit(main())
