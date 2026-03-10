###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
ROCm vs CUDA JAX/LLaMA inference trace comparison.

Consumes two Chrome-trace JSON (optionally gzipped) files from JAX inference
(ROCm and CUDA), segments by token and layer using Transformer/block_* norm_attn
markers, and emits a Markdown report with:

- Per-layer stage attribution (norm_attn, q/k/v/add+gemm, attn_core, out_gemm,
  norm_mlp, mlp_in_gemm, swiglu_elementwise, mlp_out_gemm, post_gsu, other)
- Top GPU kernels and top XLA op-paths by total time (main stream)
- Apple-to-apple stage table (µs/layer, shares, ROCm/CUDA ratios)

Assumptions: single main GPU stream; token boundaries from block_0/norm_attn;
layer boundaries from norm_attn(block_i) within each token.

Usage:
    TraceLens_compare_traces_jax_llama --rocm /path/to/rocm.trace.json.gz \\
        --cuda /path/to/cuda.trace.json.gz --tokens 1:8 --layers 0:31 --out report.md
    python -m TraceLens.Reporting.compare_traces_jax_llama --rocm ... --cuda ...
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    pid: int
    tid: int
    ts: float  # microseconds (Chrome trace)
    dur: float  # microseconds
    name: str
    args: Dict[str, Any]


@dataclass(frozen=True)
class Stats:
    count: int
    total_us: float
    avg_us: float
    p50_us: float
    p90_us: float


@dataclass
class Summary:
    label: str
    d_model: Optional[int]
    head_dim: Optional[int]
    gsu: Optional[int]
    gpu_index: int
    main_tid: int
    token_range: Tuple[int, int]
    layer_range: Tuple[int, int]
    per_layer_us: float
    per_token_us: float
    stage_avg_us: Dict[str, float]
    stage_share: Dict[str, float]
    top_kernels: List[Tuple[str, Stats]]
    top_ops: List[Tuple[str, Stats]]
    notes: List[str]


# ──────────────────────────────────────────────────────────────────────────────
# Constants / regex
# ──────────────────────────────────────────────────────────────────────────────

STAGES = [
    "norm_attn",
    "q_add",
    "q_gemm",
    "k_add",
    "k_gemm",
    "v_add",
    "v_gemm",
    "attn_core",
    "out_gemm",
    "norm_mlp",
    "mlp_in_gemm",
    "swiglu_elementwise",
    "mlp_out_gemm",
    "post_gsu",
    "other",
]

RE_BLOCK = re.compile(r"/Transformer/block_(\d+)/")
RE_TE_LN_DMODEL = re.compile(r"Kernel_traits<[^>]*,\s*(\d+)u\s*,")
RE_HEAD_DIM = re.compile(r"hd(\d+)", re.IGNORECASE)
RE_POST_GSU = re.compile(r"PostGSU(\d+)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ──────────────────────────────────────────────────────────────────────────────


def parse_range(spec: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(-?\d+)\s*:\s*(-?\d+)\s*", spec)
    if not m:
        raise ValueError(f"Bad range '{spec}', use a:b inclusive.")
    a, b = int(m.group(1)), int(m.group(2))
    if b < a:
        raise ValueError(f"Bad range '{spec}': end < start.")
    return a, b


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def mk_stats(durs: List[float]) -> Stats:
    if not durs:
        return Stats(0, 0.0, 0.0, 0.0, 0.0)
    total = float(sum(durs))
    avg = total / len(durs)
    p50 = percentile(durs, 50)
    p90 = percentile(durs, 90)
    return Stats(len(durs), total, avg, p50, p90)


def fmt_us(x: float) -> str:
    return f"{x:,.2f}"


def fmt_ms(x_us: float) -> str:
    return f"{x_us / 1000.0:,.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# Trace loading
# ──────────────────────────────────────────────────────────────────────────────


def load_trace(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def pid_map(trace: Dict[str, Any]) -> Dict[int, str]:
    mp = {}
    for e in trace["traceEvents"]:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            mp[int(e["pid"])] = str((e.get("args") or {}).get("name", ""))
    return mp


def extract_gpu_events(trace: Dict[str, Any], gpu_index: int) -> List[Event]:
    mp = pid_map(trace)
    target = f"/device:GPU:{gpu_index}"
    pids = [pid for pid, nm in mp.items() if nm == target]
    if not pids:
        pids = [pid for pid, nm in mp.items() if target in nm]
    if not pids:
        raise RuntimeError(
            f"Could not find pid for {target}. Have: {sorted(set(mp.values()))[:20]} ..."
        )
    pid = pids[0]

    out: List[Event] = []
    for e in trace["traceEvents"]:
        if e.get("ph") != "X":
            continue
        if int(e.get("pid", -1)) != pid:
            continue
        ts = e.get("ts")
        dur = e.get("dur")
        if ts is None or dur is None:
            continue
        dur = float(dur)
        if dur <= 0:
            continue
        out.append(
            Event(
                pid=pid,
                tid=int(e.get("tid", 0)),
                ts=float(ts),
                dur=dur,
                name=str(e.get("name", "")),
                args=(e.get("args") or {}),
            )
        )
    out.sort(key=lambda ev: (ev.ts, ev.tid))
    return out


def choose_main_tid(evs: List[Event]) -> int:
    by = defaultdict(float)
    for e in evs:
        by[e.tid] += e.dur
    return max(by.items(), key=lambda kv: kv[1])[0]


def get_path(e: Event) -> str:
    v = e.args.get("name")
    return str(v) if v is not None else ""


def extract_block(path: str) -> Optional[int]:
    m = RE_BLOCK.search(path)
    return int(m.group(1)) if m else None


def infer_params(
    evs: List[Event],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    d_model = None
    head_dim = None
    gsu = None
    for e in evs:
        nm = e.name
        if d_model is None and "ln_fwd_tuned_kernel" in nm:
            m = RE_TE_LN_DMODEL.search(nm)
            if m:
                d_model = int(m.group(1))
        if head_dim is None:
            m2 = RE_HEAD_DIM.search(nm)
            if m2:
                head_dim = int(m2.group(1))
        if gsu is None:
            m3 = RE_POST_GSU.search(nm)
            if m3:
                gsu = int(m3.group(1))
        if d_model and head_dim and gsu:
            break
    return d_model, head_dim, gsu


# ──────────────────────────────────────────────────────────────────────────────
# Stage classification
# ──────────────────────────────────────────────────────────────────────────────


def is_loop_multiply_fusion(e: Event) -> bool:
    if e.name.startswith("loop_multiply_fusion"):
        return True
    hlo = str(e.args.get("hlo_op", "")).lower()
    return "loop_multiply_fusion" in hlo


def classify_stage_base(e: Event) -> str:
    """
    Base stage classifier from op-path + kernel name.
    PostGSU detection by kernel name must happen before mlp/out matching.
    """
    path = get_path(e)
    kname = e.name

    if "PostGSU" in kname or "postgsu" in kname.lower():
        return "post_gsu"

    if "/norm_attn/" in path:
        return "norm_attn"
    if "/norm_mlp/" in path:
        return "norm_mlp"

    if "/attn/q/add" in path:
        return "q_add"
    if "/attn/k/add" in path:
        return "k_add"
    if "/attn/v/add" in path:
        return "v_add"

    if "/attn/q/dot_general" in path:
        return "q_gemm"
    if "/attn/k/dot_general" in path:
        return "k_gemm"
    if "/attn/v/dot_general" in path:
        return "v_gemm"
    if "/attn/out/dot_general" in path:
        return "out_gemm"
    if "/mlp/in/dot_general" in path:
        return "mlp_in_gemm"
    if "/mlp/out/dot_general" in path:
        return "mlp_out_gemm"

    if "te_fused_attn_forward" in path:
        return "attn_core"
    if "sdpa" in kname.lower() or "flash_fprop" in kname.lower():
        return "attn_core"
    if "fmha_fwd" in kname.lower():
        return "attn_core"

    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Token & layer segmentation
# ──────────────────────────────────────────────────────────────────────────────


def token_start_times(
    stream: List[Event], block0_norm_hint: str
) -> List[float]:
    """
    Token boundaries: first event in each token is block_0/norm_attn/...
    ROCm path contains te_layernorm_forward; CUDA path contains te_norm_forward_ffi.
    """
    starts = []
    for e in stream:
        p = get_path(e)
        if "/Transformer/block_0/norm_attn/" in p and block0_norm_hint in p:
            starts.append(e.ts)
    return sorted(starts)


def collect_norm_attn_times_for_token(
    stream: List[Event], t_start: float, t_end: float
) -> Dict[int, float]:
    """Within a token window, collect the first norm_attn timestamp per block."""
    times: Dict[int, float] = {}
    for e in stream:
        if e.ts < t_start:
            continue
        if e.ts >= t_end:
            break
        p = get_path(e)
        if "/norm_attn/" not in p:
            continue
        b = extract_block(p)
        if b is None:
            continue
        if b not in times:
            times[b] = e.ts
    return times


def build_ts_index(stream: List[Event]) -> List[float]:
    return [e.ts for e in stream]


def slice_by_time(
    stream: List[Event], ts_index: List[float], a: float, b: float
) -> List[Event]:
    """Return events with ts in [a, b). stream is sorted; ts_index is list of stream.ts."""
    ia = bisect_left(ts_index, a)
    ib = bisect_left(ts_index, b)
    return stream[ia:ib]


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer stage accounting (SwiGLU attribution)
# ──────────────────────────────────────────────────────────────────────────────


def compute_stage_table(
    stream: List[Event],
    token_starts: List[float],
    token_range: Tuple[int, int],
    layer_range: Tuple[int, int],
) -> Tuple[
    Dict[str, float], Dict[str, float], float, float, List[str]
]:
    """
    Returns stage_avg_us per layer, stage_share, per_layer_us, per_token_us, notes.

    loop_multiply_fusion* events with args.name containing .../Transformer (no block)
    are attributed to swiglu_elementwise when they occur between mlp/in and mlp/out
    within the same layer interval.
    """
    t0, t1 = token_range
    l0, l1 = layer_range

    notes: List[str] = []

    ts_index = build_ts_index(stream)
    max_ts_end = stream[-1].ts + stream[-1].dur

    stage_sum = defaultdict(float)
    total_layers = 0
    per_token_totals: List[float] = []

    for tok in range(t0, t1 + 1):
        if tok >= len(token_starts):
            break
        t_start = token_starts[tok]
        t_end = (
            token_starts[tok + 1]
            if (tok + 1) < len(token_starts)
            else max_ts_end
        )

        norm_times = collect_norm_attn_times_for_token(
            stream, t_start, t_end
        )

        need_blocks = list(range(l0, l1 + 1))
        if any(b not in norm_times for b in need_blocks):
            notes.append(
                f"token {tok}: incomplete blocks (have {len(norm_times)}); "
                "skipped in stage avg"
            )
            continue

        token_total = 0.0

        for b in range(l0, l1 + 1):
            layer_start = norm_times[b]
            layer_end = norm_times.get(b + 1, t_end)

            layer_events = slice_by_time(
                stream, ts_index, layer_start, layer_end
            )

            mlp_in_ev = None
            mlp_out_ev = None
            for e in layer_events:
                p = get_path(e)
                if mlp_in_ev is None and "/mlp/in/dot_general" in p:
                    mlp_in_ev = e
                if mlp_out_ev is None and "/mlp/out/dot_general" in p:
                    mlp_out_ev = e

            mlp_in_end = (
                (mlp_in_ev.ts + mlp_in_ev.dur) if mlp_in_ev else None
            )
            mlp_out_start = mlp_out_ev.ts if mlp_out_ev else None

            layer_stage = defaultdict(float)
            layer_total = 0.0

            for e in layer_events:
                st = classify_stage_base(e)

                if (
                    st == "other"
                    and is_loop_multiply_fusion(e)
                    and mlp_in_end is not None
                    and mlp_out_start is not None
                ):
                    if mlp_in_end <= e.ts < mlp_out_start:
                        st = "swiglu_elementwise"

                layer_stage[st] += e.dur
                layer_total += e.dur

            for s, v in layer_stage.items():
                stage_sum[s] += v

            token_total += layer_total
            total_layers += 1

        per_token_totals.append(token_total)

    if total_layers == 0:
        raise RuntimeError(
            "No complete token/layer windows found. "
            "Check token/layer range and norm_attn markers."
        )

    stage_avg = {
        s: stage_sum.get(s, 0.0) / total_layers for s in STAGES
    }
    per_layer = sum(stage_avg.values())
    stage_share = {
        s: (stage_avg[s] / per_layer if per_layer > 0 else 0.0)
        for s in STAGES
    }
    per_token = (
        statistics.mean(per_token_totals) if per_token_totals else float("nan")
    )

    return stage_avg, stage_share, per_layer, per_token, notes


# ──────────────────────────────────────────────────────────────────────────────
# Top kernels / top op-paths (main stream)
# ──────────────────────────────────────────────────────────────────────────────


def top_stats_by_key(
    stream: List[Event], key_fn, top_n: int
) -> List[Tuple[str, Stats]]:
    by = defaultdict(list)
    for e in stream:
        k = key_fn(e)
        if not k:
            continue
        by[k].append(e.dur)
    stats = [(k, mk_stats(durs)) for k, durs in by.items()]
    stats.sort(key=lambda kv: kv[1].total_us, reverse=True)
    return stats[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# Build summary
# ──────────────────────────────────────────────────────────────────────────────


def summarize_one(
    label: str,
    trace_path: str,
    gpu_index: int,
    tokens: Tuple[int, int],
    layers: Tuple[int, int],
    block0_norm_hint: str,
    top_kernels_n: int = 12,
    top_ops_n: int = 15,
) -> Summary:
    trace = load_trace(trace_path)
    gpu_events = extract_gpu_events(trace, gpu_index=gpu_index)
    main_tid = choose_main_tid(gpu_events)
    stream = [e for e in gpu_events if e.tid == main_tid]
    stream.sort(key=lambda e: e.ts)

    d_model, head_dim, gsu = infer_params(stream)

    token_starts = token_start_times(
        stream, block0_norm_hint=block0_norm_hint
    )

    stage_avg, stage_share, per_layer, per_token, notes = (
        compute_stage_table(
            stream=stream,
            token_starts=token_starts,
            token_range=tokens,
            layer_range=layers,
        )
    )

    top_kernels = top_stats_by_key(
        stream, key_fn=lambda e: e.name, top_n=top_kernels_n
    )
    top_ops = top_stats_by_key(
        stream, key_fn=lambda e: get_path(e), top_n=top_ops_n
    )

    return Summary(
        label=label,
        d_model=d_model,
        head_dim=head_dim,
        gsu=gsu,
        gpu_index=gpu_index,
        main_tid=main_tid,
        token_range=tokens,
        layer_range=layers,
        per_layer_us=per_layer,
        per_token_us=per_token,
        stage_avg_us=stage_avg,
        stage_share=stage_share,
        top_kernels=top_kernels,
        top_ops=top_ops,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────


def emit_report(rocm: Summary, cuda: Summary) -> str:
    lines: List[str] = []
    lines.append(
        "# Trace Comparison: ROCm vs CUDA (full + fixed attribution)\n\n"
    )

    lines.append("## Workload sanity checks\n\n")
    lines.append(
        f"- ROCm: d_model={rocm.d_model}, head_dim={rocm.head_dim}, "
        f"PostGSU={rocm.gsu}\n"
    )
    lines.append(
        f"- CUDA: d_model={cuda.d_model}, head_dim={cuda.head_dim}, "
        f"PostGSU={cuda.gsu}\n"
    )
    lines.append(
        f"- Token range analyzed: ROCm {rocm.token_range[0]}:{rocm.token_range[1]}, "
        f"CUDA {cuda.token_range[0]}:{cuda.token_range[1]}\n"
    )
    lines.append(
        f"- Layer range analyzed: ROCm {rocm.layer_range[0]}:{rocm.layer_range[1]}, "
        f"CUDA {cuda.layer_range[0]}:{cuda.layer_range[1]}\n\n"
    )

    ratio = (
        (rocm.per_layer_us / cuda.per_layer_us)
        if cuda.per_layer_us > 0
        else float("inf")
    )
    lines.append("## Critical-path timing (GPU main stream)\n\n")
    lines.append(
        f"- ROCm avg: **{fmt_ms(rocm.per_token_us)} ms/token** "
        f"({fmt_us(rocm.per_layer_us)} µs/layer)\n"
    )
    lines.append(
        f"- CUDA avg: **{fmt_ms(cuda.per_token_us)} ms/token** "
        f"({fmt_us(cuda.per_layer_us)} µs/layer)\n"
    )
    lines.append(
        f"- ROCm/CUDA per-layer ratio: **{ratio:.3f}×** "
        "( >1 means ROCm slower )\n\n"
    )

    lines.append(
        "## Per-layer time (µs) and share of layer critical path\n\n"
    )
    lines.append(
        "| Stage | ROCm µs/layer | ROCm share | CUDA µs/layer | "
        "CUDA share | ROCm/CUDA |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for s in STAGES:
        r_us = rocm.stage_avg_us.get(s, 0.0)
        c_us = cuda.stage_avg_us.get(s, 0.0)
        r_sh = rocm.stage_share.get(s, 0.0) * 100.0
        c_sh = cuda.stage_share.get(s, 0.0) * 100.0
        rc = (r_us / c_us) if c_us > 0 else float("inf")
        lines.append(
            f"| {s} | {fmt_us(r_us)} | {r_sh:,.2f}% | {fmt_us(c_us)} | "
            f"{c_sh:,.2f}% | {rc:.3f} |\n"
        )

    lines.append("\n## Top GPU kernels by total time (main stream)\n\n")
    lines.append("### ROCm (top 12)\n")
    for name, st in rocm.top_kernels[:12]:
        lines.append(
            f"- `{name}` — count={st.count}, total={fmt_ms(st.total_us)} ms, "
            f"avg={fmt_us(st.avg_us)} µs, p90={fmt_us(st.p90_us)} µs\n"
        )
    lines.append("\n### CUDA (top 12)\n")
    for name, st in cuda.top_kernels[:12]:
        lines.append(
            f"- `{name}` — count={st.count}, total={fmt_ms(st.total_us)} ms, "
            f"avg={fmt_us(st.avg_us)} µs, p90={fmt_us(st.p90_us)} µs\n"
        )

    lines.append(
        "\n## Top XLA op-paths by total time (main stream)\n"
    )
    lines.append(
        "These come from `event.args.name` "
        "(e.g., `.../attn/q/dot_general`).\n\n"
    )
    lines.append("### ROCm (top 15)\n")
    for path, st in rocm.top_ops[:15]:
        lines.append(
            f"- `{path}` — count={st.count}, total={fmt_ms(st.total_us)} ms, "
            f"avg={fmt_us(st.avg_us)} µs\n"
        )
    lines.append("\n### CUDA (top 15)\n")
    for path, st in cuda.top_ops[:15]:
        lines.append(
            f"- `{path}` — count={st.count}, total={fmt_ms(st.total_us)} ms, "
            f"avg={fmt_us(st.avg_us)} µs\n"
        )

    lines.append(
        "\n## Interpretation notes (how to read differences)\n"
    )
    lines.append(
        "- If ROCm shows non-zero `post_gsu` and CUDA shows ~0, "
        "ROCm is paying split-K/GSU reduction overhead "
        "(extra bandwidth pass + launch).\n"
    )
    lines.append(
        "- If CUDA shows non-zero `swiglu_elementwise` but ROCm shows ~0, "
        "the SwiGLU gating is materializing as an explicit kernel on CUDA "
        "but may be fused/hidden or lowered differently on ROCm.\n"
    )
    lines.append(
        "- Biggest ratios usually appear in q/k/v/out projections in decode "
        "because M is tiny (batch*tokens), making GEMM kernel selection "
        "and small-M optimizations decisive.\n"
    )

    if rocm.notes or cuda.notes:
        lines.append("\n## Notes\n")
        for n in rocm.notes:
            lines.append(f"- [ROCm] {n}\n")
        for n in cuda.notes:
            lines.append(f"- [CUDA] {n}\n")

    return "".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ROCm vs CUDA JAX/LLaMA inference traces; "
        "output Markdown report with per-layer stage breakdown and top kernels."
    )
    parser.add_argument(
        "--rocm",
        type=str,
        required=True,
        help="Path to ROCm trace (Chrome JSON, optionally .gz)",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        required=True,
        help="Path to CUDA trace (Chrome JSON, optionally .gz)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index used in trace process names (default: 0)",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="1:8",
        help="Token range to analyze, inclusive (default: 1:8)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0:31",
        help="Layer range to analyze, inclusive (default: 0:31)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output Markdown file; if omitted, print to stdout",
    )
    args = parser.parse_args()

    tokens = parse_range(args.tokens)
    layers = parse_range(args.layers)

    rocm_hint = "te_layernorm_forward"
    cuda_hint = "te_norm_forward_ffi"

    rocm = summarize_one(
        "ROCm",
        args.rocm,
        args.gpu,
        tokens,
        layers,
        rocm_hint,
        top_kernels_n=12,
        top_ops_n=15,
    )
    cuda = summarize_one(
        "CUDA",
        args.cuda,
        args.gpu,
        tokens,
        layers,
        cuda_hint,
        top_kernels_n=12,
        top_ops_n=15,
    )

    report = emit_report(rocm, cuda)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
