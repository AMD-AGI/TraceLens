###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Perfetto-style trace (e.g. rocprofv3 pftrace) activity analysis: GPU category
summary, kernel summary (by config or name), HIP API summary, XLA top kernels.
Shared logic ported from pftrace_hip_api_hip_activity_report; avoids duplication
with pftrace_hip_api_analysis (API↔kernel correlation) by focusing on
per-GPU event lists and NSYS-style summaries.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data models (match hip_activity_report for compatibility)                   #
# --------------------------------------------------------------------------- #


@dataclass
class Event:
    gpu: int
    name: str
    ts_ns: int
    dur_ns: int
    tid: int = -1
    grid_size: Optional[int] = None
    workgroup_size: Optional[int] = None
    vgpr_count: Optional[int] = None
    accum_vgpr_count: Optional[int] = None
    sgpr_count: Optional[int] = None
    lds_block_size: Optional[int] = None
    scratch_size: Optional[int] = None
    stream_id: Optional[int] = None
    queue_id: Optional[int] = None
    corr_id: Optional[int] = None
    kernel_id: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HIPEvent:
    name: str
    ts_ns: int
    dur_ns: int
    pid: int
    tid: int
    stream_id: Optional[int] = None
    operation: Optional[int] = None
    corr_id: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUStats:
    xla: Tuple[int, int] = (0, 0)
    aiter_fwd: Tuple[int, int] = (0, 0)
    aiter_bwd: Tuple[int, int] = (0, 0)
    gemm: Tuple[int, int] = (0, 0)
    ck_fwd: Tuple[int, int] = (0, 0)
    ck_bwd: Tuple[int, int] = (0, 0)
    rccl: Tuple[int, int] = (0, 0)
    te: Tuple[int, int] = (0, 0)
    memcpy: Tuple[int, int] = (0, 0)
    fillBuffer: Tuple[int, int] = (0, 0)
    overlapped_comm_ns: int = 0
    non_overlapped_comm_ns: int = 0
    compute_total_ns: int = 0


# --------------------------------------------------------------------------- #
# Time and GPU discovery                                                       #
# --------------------------------------------------------------------------- #


def extract_time_ns(e: Dict[str, Any]) -> Tuple[int, int]:
    """
    Robustly extract (ts_ns, dur_ns) from a Perfetto event.
    Prefer args.begin_ns/delta_ns; else treat ts/dur as microseconds and convert.
    """
    args = e.get("args") or {}
    if "begin_ns" in args and "delta_ns" in args:
        ts_ns = int(args["begin_ns"])
        dur_ns = int(args["delta_ns"])
    else:
        ts = int(e.get("ts", 0))
        dur = int(e.get("dur", 0))
        if ts < 10_000_000_000:
            ts_ns = ts * 1000
            dur_ns = dur * 1000
        else:
            ts_ns = ts
            dur_ns = dur
    return ts_ns, dur_ns


def discover_gpus(trace_events: List[Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    """Map 'agent' strings to 0..N-1 GPU indices in first-seen order."""
    agent_to_idx: Dict[str, int] = {}
    agents: List[str] = []
    for ev in trace_events:
        agent = (ev.get("args") or {}).get("agent")
        if not agent:
            continue
        if agent not in agent_to_idx:
            agent_to_idx[agent] = len(agents)
            agents.append(agent)
    return agent_to_idx, agents


# --------------------------------------------------------------------------- #
# Classification (kernel names → category)                                    #
# --------------------------------------------------------------------------- #

_c_gemm = re.compile(r"(cublasLt|Cijk|gemm|nvjet)", re.IGNORECASE)
_c_rccl = re.compile(r"(rccl|nccl)", re.IGNORECASE)
_c_mem = re.compile(r"(copy|memcpy|memset)", re.IGNORECASE)
_c_ck_fwd = re.compile(r"(FmhaFwd|flash_fprop)", re.IGNORECASE)
_c_ck_bwd = re.compile(r"(FmhaBwd|kernel_func|flash_bprop)", re.IGNORECASE)
_c_te = re.compile(r"(transformer_engine)", re.IGNORECASE)
_c_aiter_fwd = re.compile(r"(aiter::fmha_fwd)", re.IGNORECASE)
_c_aiter_bwd = re.compile(r"(aiter::fmha_bwd)", re.IGNORECASE)
_c_fillBuffer = re.compile(r"(fillbuffer)", re.IGNORECASE)


def classify(name: str) -> str:
    if _c_rccl.search(name):
        return "rccl"
    if _c_gemm.search(name):
        return "gemm"
    if _c_ck_bwd.search(name):
        return "ckbwd"
    if _c_ck_fwd.search(name):
        return "ckfwd"
    if _c_mem.search(name):
        return "memcpy"
    if _c_te.search(name):
        return "te"
    if _c_aiter_fwd.search(name):
        return "aiterfwd"
    if _c_aiter_bwd.search(name):
        return "aiterbwd"
    if _c_fillBuffer.search(name):
        return "fillBuffer"
    return "xla"


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Event list building                                                         #
# --------------------------------------------------------------------------- #


def build_event_lists(
    trace_events: List[Dict[str, Any]],
    merge_kernels: bool,
    min_tid: int,
    max_tid: int,
) -> Tuple[
    List[List[Event]],
    List[List[Event]],
    Dict[str, List[int]],
    bool,
    List[str],
]:
    """
    Returns:
      compute_events[g]: non-RCCL Event list for GPU g (sorted)
      rccl_events[g]: RCCL Event list for GPU g (sorted)
      xla_agg: dict name -> [total_ns, count] across GPUs
      used_fav3: bool
      agents: agent names in GPU index order
    """
    agent_to_idx, agents = discover_gpus(trace_events)
    n = len(agents)
    compute_events: List[List[Event]] = [[] for _ in range(n)]
    rccl_events: List[List[Event]] = [[] for _ in range(n)]
    xla_agg: Dict[str, List[int]] = {}
    used_fav3 = False

    for ev in trace_events:
        args = ev.get("args") or {}
        agent = args.get("agent")
        if not agent or agent not in agent_to_idx:
            continue
        gpu_idx = agent_to_idx[agent]
        ts_ns, dur_ns = extract_time_ns(ev)
        if dur_ns <= 0:
            continue
        tid = int(ev.get("tid", -1))
        if tid < min_tid or tid > max_tid:
            continue
        name = str(ev.get("name", ""))
        cat = classify(name)

        e = Event(
            gpu=gpu_idx,
            name=name,
            ts_ns=ts_ns,
            dur_ns=dur_ns,
            tid=tid,
            grid_size=_safe_int(args.get("grid_size")),
            workgroup_size=_safe_int(args.get("workgroup_size")),
            vgpr_count=_safe_int(args.get("VGPR_Count")),
            accum_vgpr_count=_safe_int(args.get("Accum_VGPR_Count")),
            sgpr_count=_safe_int(args.get("SGPR_Count")),
            lds_block_size=_safe_int(args.get("LDS_Block_Size")),
            scratch_size=_safe_int(args.get("Scratch_Size")),
            stream_id=_safe_int(args.get("stream_ID")),
            queue_id=_safe_int(args.get("queue")),
            corr_id=_safe_int(args.get("corr_id")),
            kernel_id=_safe_int(args.get("kernel_id")),
            raw=ev,
        )
        if cat == "rccl":
            rccl_events[gpu_idx].append(e)
        else:
            compute_events[gpu_idx].append(e)
            if cat == "ckbwd" and "kernel_func" in name:
                used_fav3 = True
            if cat == "xla":
                key = re.sub(r"\d+", "", name) if merge_kernels else name
                acc = xla_agg.setdefault(key, [0, 0])
                acc[0] += dur_ns
                acc[1] += 1

    for g in range(n):
        compute_events[g].sort(key=lambda e: (e.ts_ns, e.dur_ns, e.name))
        rccl_events[g].sort(key=lambda e: (e.ts_ns, e.dur_ns, e.name))
    return compute_events, rccl_events, xla_agg, used_fav3, agents


def build_hip_api_events(
    trace_events: List[Dict[str, Any]],
    min_tid: int,
    max_tid: int,
) -> List[HIPEvent]:
    hip_events: List[HIPEvent] = []
    for ev in trace_events:
        if (ev.get("cat") or "") != "hip_api":
            continue
        args = ev.get("args") or {}
        ts_ns, dur_ns = extract_time_ns(ev)
        if dur_ns <= 0:
            continue
        tid = int(ev.get("tid", args.get("tid", -1)))
        if tid < min_tid or tid > max_tid:
            continue
        hip_events.append(
            HIPEvent(
                name=str(ev.get("name", "")),
                ts_ns=ts_ns,
                dur_ns=dur_ns,
                pid=int(ev.get("pid", -1)),
                tid=tid,
                stream_id=_safe_int(args.get("stream_ID")),
                operation=_safe_int(args.get("operation")),
                corr_id=_safe_int(args.get("corr_id")),
                raw=ev,
            )
        )
    hip_events.sort(key=lambda e: (e.ts_ns, e.dur_ns, e.name))
    return hip_events


# --------------------------------------------------------------------------- #
# Category accumulation and RCCL overlap                                      #
# --------------------------------------------------------------------------- #


def _sum_tuple(t: Tuple[int, int], add_ns: int, add_cnt: int) -> Tuple[int, int]:
    return (t[0] + add_ns, t[1] + add_cnt)


def accumulate_categories(events: Iterable[Event], stats: GPUStats) -> GPUStats:
    for e in events:
        cat = classify(e.name)
        if cat == "gemm":
            stats.gemm = _sum_tuple(stats.gemm, e.dur_ns, 1)
        elif cat == "ckbwd":
            stats.ck_bwd = _sum_tuple(stats.ck_bwd, e.dur_ns, 1)
        elif cat == "ckfwd":
            stats.ck_fwd = _sum_tuple(stats.ck_fwd, e.dur_ns, 1)
        elif cat == "memcpy":
            stats.memcpy = _sum_tuple(stats.memcpy, e.dur_ns, 1)
        elif cat == "te":
            stats.te = _sum_tuple(stats.te, e.dur_ns, 1)
        elif cat == "aiterfwd":
            stats.aiter_fwd = _sum_tuple(stats.aiter_fwd, e.dur_ns, 1)
        elif cat == "aiterbwd":
            stats.aiter_bwd = _sum_tuple(stats.aiter_bwd, e.dur_ns, 1)
        elif cat == "rccl":
            stats.rccl = _sum_tuple(stats.rccl, e.dur_ns, 1)
        elif cat == "fillBuffer":
            stats.fillBuffer = _sum_tuple(stats.fillBuffer, e.dur_ns, 1)
        else:
            stats.xla = _sum_tuple(stats.xla, e.dur_ns, 1)
    return stats


def rccl_overlap_two_pointer(
    compute: List[Event], comm: List[Event]
) -> Tuple[int, int]:
    """Returns (overlapped_ns, non_overlapped_ns) for RCCL vs compute. Lists sorted by ts_ns."""
    i = 0
    overlapped = 0
    comm_total = 0
    for c in comm:
        c_start, c_end = c.ts_ns, c.ts_ns + c.dur_ns
        comm_total += c.dur_ns
        while i < len(compute) and (compute[i].ts_ns + compute[i].dur_ns) <= c_start:
            i += 1
        k = i
        while k < len(compute) and compute[k].ts_ns < c_end:
            s1, e1 = compute[k].ts_ns, compute[k].ts_ns + compute[k].dur_ns
            ov = max(0, min(e1, c_end) - max(s1, c_start))
            overlapped += ov
            if e1 <= c_end:
                k += 1
            else:
                break
    return overlapped, comm_total - overlapped


# --------------------------------------------------------------------------- #
# Reporting helpers                                                           #
# --------------------------------------------------------------------------- #


def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def human_time_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.3f} μs"
    return f"{int(ns)} ns"


def build_summary_dataframe(
    gpu_stats: List[GPUStats],
    label_compute_fraction: str = "Fraction of compute runtime",
    label_total_fraction: str = "Fraction of total runtime",
) -> pd.DataFrame:
    rows = []
    for g, st in enumerate(gpu_stats):
        compute_ns = (
            st.xla[0] + st.aiter_fwd[0] + st.aiter_bwd[0]
            + st.gemm[0] + st.ck_bwd[0] + st.ck_fwd[0]
            + st.te[0] + st.memcpy[0] + st.fillBuffer[0]
        )
        total_ns = compute_ns + st.rccl[0]
        per_cat = {
            "xla": st.xla,
            "aiterfwd": st.aiter_fwd,
            "aiterbwd": st.aiter_bwd,
            "gemm": st.gemm,
            "ckbwd": st.ck_bwd,
            "ckfwd": st.ck_fwd,
            "te": st.te,
            "memcpy/memset": st.memcpy,
            "fillBuffer": st.fillBuffer,
            "rccl": st.rccl,
            "overlapped_comm": (st.overlapped_comm_ns, 0),
            "non_overlapped_comm": (st.non_overlapped_comm_ns, 0),
            "compute_total": (compute_ns, 0),
            "total_incl_comm": (total_ns, 0),
        }
        for cat, (ns_sum, cnt) in per_cat.items():
            rows.append({
                "GPU ID": g,
                "Category": cat,
                "Total runtime (ms)": ns_to_ms(ns_sum),
                "Total count": cnt,
                label_compute_fraction: (ns_sum / compute_ns) if compute_ns > 0 else 0.0,
                label_total_fraction: (ns_sum / total_ns) if total_ns > 0 else 0.0,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        avg_rows = []
        for cat, grp in df.groupby("Category"):
            avg_rows.append({
                "GPU ID": "Avg",
                "Category": cat,
                "Total runtime (ms)": grp["Total runtime (ms)"].mean(),
                "Total count": grp["Total count"].mean(),
                label_compute_fraction: grp[label_compute_fraction].mean(),
                label_total_fraction: grp[label_total_fraction].mean(),
            })
        df = pd.concat([df, pd.DataFrame(avg_rows)], ignore_index=True)
    return df


def _nz(x: Optional[int]) -> int:
    return int(x) if isinstance(x, (int, np.integer)) and x is not None else (x if x else 0)


def build_kernel_summary_df_for_config(
    events: Iterable[Event],
    baseline_total_ns: int,
    merge_names: bool,
) -> pd.DataFrame:
    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    meta_rows: Dict[Tuple, Dict[str, Any]] = {}
    for e in events:
        name = re.sub(r"\d+", "", e.name) if merge_names else e.name
        key = (
            name,
            _nz(e.grid_size), _nz(e.workgroup_size),
            _nz(e.vgpr_count), _nz(e.accum_vgpr_count),
            _nz(e.sgpr_count), _nz(e.lds_block_size), _nz(e.scratch_size),
        )
        buckets[key].append(e.dur_ns)
        if key not in meta_rows:
            meta_rows[key] = {
                "Name": name,
                "GridX": _nz(e.grid_size), "GridY": 1, "GridZ": 1,
                "BlockX": _nz(e.workgroup_size), "BlockY": 1, "BlockZ": 1,
                "VGPR": _nz(e.vgpr_count),
                "AccumVGPR": _nz(e.accum_vgpr_count),
                "SGPR": _nz(e.sgpr_count),
                "LDS": _nz(e.lds_block_size),
                "Scratch": _nz(e.scratch_size),
            }
    rows = []
    for key, durs in buckets.items():
        arr = np.array(durs, dtype=np.int64)
        total_ns = int(arr.sum())
        count = int(arr.size)
        avg_ns = float(arr.mean()) if count else 0.0
        med_ns = float(np.median(arr)) if count else 0.0
        min_ns = int(arr.min()) if count else 0
        max_ns = int(arr.max()) if count else 0
        std_ns = float(arr.std(ddof=0)) if count else 0.0
        frac = (total_ns / baseline_total_ns) if baseline_total_ns > 0 else 0.0
        meta = meta_rows[key]
        rows.append({
            "Time %": frac * 100.0,
            "Total Time (ns)": total_ns,
            "Instances": count,
            "Avg (ns)": avg_ns,
            "Med (ns)": med_ns,
            "Min (ns)": min_ns,
            "Max (ns)": max_ns,
            "StdDev (ns)": std_ns,
            **meta,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("Total Time (ns)", ascending=False, inplace=True)
    df["Time"] = df["Time %"].map(lambda p: f"{p:.1f}%")
    df["Total Time"] = df["Total Time (ns)"].map(human_time_ns)
    for col in ["Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]:
        pretty = col.replace("(ns)", "").strip()
        df[pretty] = df[col].map(human_time_ns)
    df["GridXYZ"] = df.apply(lambda r: f"{r['GridX']:>7d} {r['GridY']:>4d} {r['GridZ']:>4d}", axis=1)
    df["BlockXYZ"] = df.apply(lambda r: f"{r['BlockX']:>7d} {r['BlockY']:>4d} {r['BlockZ']:>4d}", axis=1)
    col_order = [
        "Time", "Total Time", "Instances", "Avg", "Med", "Min", "Max", "StdDev",
        "GridXYZ", "BlockXYZ", "VGPR", "AccumVGPR", "SGPR", "LDS", "Scratch", "Name",
        "Time %", "Total Time (ns)", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)",
        "GridX", "GridY", "GridZ", "BlockX", "BlockY", "BlockZ",
    ]
    return df[[c for c in col_order if c in df.columns]]


def build_kernel_summary_df_for_name(
    events: Iterable[Event],
    baseline_total_ns: int,
    merge_names: bool,
) -> pd.DataFrame:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for e in events:
        name = re.sub(r"\d+", "", e.name) if merge_names else e.name
        buckets[name].append(e.dur_ns)
    rows = []
    for name, durs in buckets.items():
        arr = np.array(durs, dtype=np.int64)
        total_ns = int(arr.sum())
        count = int(arr.size)
        avg_ns = float(arr.mean()) if count else 0.0
        med_ns = float(np.median(arr)) if count else 0.0
        min_ns = int(arr.min()) if count else 0
        max_ns = int(arr.max()) if count else 0
        std_ns = float(arr.std(ddof=0)) if count else 0.0
        frac = (total_ns / baseline_total_ns) if baseline_total_ns > 0 else 0.0
        rows.append({
            "Time %": frac * 100.0,
            "Total Time (ns)": total_ns,
            "Instances": count,
            "Avg (ns)": avg_ns,
            "Med (ns)": med_ns,
            "Min (ns)": min_ns,
            "Max (ns)": max_ns,
            "StdDev (ns)": std_ns,
            "Name": name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("Total Time (ns)", ascending=False, inplace=True)
    df["Time"] = df["Time %"].map(lambda p: f"{p:.1f}%")
    df["Total Time"] = df["Total Time (ns)"].map(human_time_ns)
    for col in ["Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]:
        pretty = col.replace("(ns)", "").strip()
        df[pretty] = df[col].map(human_time_ns)
    col_order = [
        "Time", "Total Time", "Instances", "Avg", "Med", "Min", "Max", "StdDev", "Name",
        "Time %", "Total Time (ns)", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)",
    ]
    return df[[c for c in col_order if c in df.columns]]


def build_hip_summary_df(
    hip_events: Iterable[HIPEvent],
    group: str = "name",
) -> pd.DataFrame:
    """Build NSYS-style summary over HIP API calls. group in name, name+stream, name+op, name+stream+op."""
    if not hip_events:
        return pd.DataFrame()
    hip_list = list(hip_events)
    baseline_total_ns = int(sum(e.dur_ns for e in hip_list))

    def key_of(e: HIPEvent):
        if group == "name":
            return (e.name,)
        if group == "name+stream":
            return (e.name, e.stream_id or 0)
        if group == "name+op":
            return (e.name, e.operation or 0)
        return (e.name, e.stream_id or 0, e.operation or 0)

    buckets: Dict[Tuple, List[int]] = defaultdict(list)
    meta: Dict[Tuple, Dict[str, Any]] = {}
    for e in hip_list:
        k = key_of(e)
        buckets[k].append(e.dur_ns)
        if k not in meta:
            row = {"Name": e.name}
            if group in ("name+stream", "name+stream+op"):
                row["Stream"] = int(e.stream_id or 0)
            if group in ("name+op", "name+stream+op"):
                row["Op"] = int(e.operation or 0)
            meta[k] = row

    rows = []
    for k, durs in buckets.items():
        arr = np.array(durs, dtype=np.int64)
        total_ns = int(arr.sum())
        n = int(arr.size)
        avg_ns = float(arr.mean()) if n else 0.0
        med_ns = float(np.median(arr)) if n else 0.0
        min_ns = int(arr.min()) if n else 0
        max_ns = int(arr.max()) if n else 0
        std_ns = float(arr.std(ddof=0)) if n else 0.0
        frac = (total_ns / baseline_total_ns) if baseline_total_ns > 0 else 0.0
        rows.append({
            "Time %": frac * 100.0,
            "Total Time (ns)": total_ns,
            "Instances": n,
            "Avg (ns)": avg_ns,
            "Med (ns)": med_ns,
            "Min (ns)": min_ns,
            "Max (ns)": max_ns,
            "StdDev (ns)": std_ns,
            **meta[k],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values("Total Time (ns)", ascending=False, inplace=True)
    df["Time"] = df["Time %"].map(lambda p: f"{p:.1f}%")
    df["Total Time"] = df["Total Time (ns)"].map(human_time_ns)
    for col in ["Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]:
        pretty = col.replace("(ns)", "").strip()
        df[pretty] = df[col].map(human_time_ns)
    base_cols = ["Time", "Total Time", "Instances", "Avg", "Med", "Min", "Max", "StdDev"]
    if group == "name":
        order = base_cols + ["Name"]
    elif group == "name+stream":
        order = base_cols + ["Stream", "Name"]
    elif group == "name+op":
        order = base_cols + ["Op", "Name"]
    else:
        order = base_cols + ["Stream", "Op", "Name"]
    order += ["Time %", "Total Time (ns)", "Avg (ns)", "Med (ns)", "Min (ns)", "Max (ns)", "StdDev (ns)"]
    return df[[c for c in order if c in df.columns]]


# --------------------------------------------------------------------------- #
# High-level analyzer (single entry for report generator)                     #
# --------------------------------------------------------------------------- #


class PftraceHipActivityAnalyzer:
    """
    Runs full hip-activity pipeline: event lists, per-GPU stats, category summary,
    kernel summary, HIP API summary, XLA top. Use get_* methods for DataFrames.
    """

    def __init__(
        self,
        trace_events: List[Dict[str, Any]],
        merge_kernels: bool = False,
        min_tid: int = -10**9,
        max_tid: int = 10**9,
        min_event_ns: int = 5000,
        kernel_summary_include_rccl: bool = False,
        kernel_summary_baseline: str = "total",
        kernel_summary_group: str = "config",
        hip_summary_group: str = "name",
    ):
        self.trace_events = trace_events
        self.merge_kernels = merge_kernels
        self.min_tid = min_tid
        self.max_tid = max_tid
        self.min_event_ns = min_event_ns
        self.kernel_summary_include_rccl = kernel_summary_include_rccl
        self.kernel_summary_baseline = kernel_summary_baseline
        self.kernel_summary_group = kernel_summary_group
        self.hip_summary_group = hip_summary_group

        self._compute_events: List[List[Event]] = []
        self._rccl_events: List[List[Event]] = []
        self._xla_agg: Dict[str, List[int]] = {}
        self._used_fav3 = False
        self._agents: List[str] = []
        self._gpu_stats: List[GPUStats] = []
        self._hip_events: List[HIPEvent] = []

        self._run()

    def _run(self) -> None:
        compute_events, rccl_events, xla_agg, used_fav3, agents = build_event_lists(
            self.trace_events,
            merge_kernels=self.merge_kernels,
            min_tid=self.min_tid,
            max_tid=self.max_tid,
        )
        self._compute_events = compute_events
        self._rccl_events = rccl_events
        self._xla_agg = xla_agg
        self._used_fav3 = used_fav3
        self._agents = agents
        num_gpus = len(agents)
        self._gpu_stats = [GPUStats() for _ in range(num_gpus)]

        for g in range(num_gpus):
            comp = [e for e in compute_events[g] if e.dur_ns >= self.min_event_ns]
            comm = [e for e in rccl_events[g] if e.dur_ns >= self.min_event_ns]
            self._gpu_stats[g] = accumulate_categories(comp, self._gpu_stats[g])
            self._gpu_stats[g] = accumulate_categories(comm, self._gpu_stats[g])
            self._gpu_stats[g].compute_total_ns = (
                self._gpu_stats[g].xla[0] + self._gpu_stats[g].aiter_fwd[0] + self._gpu_stats[g].aiter_bwd[0]
                + self._gpu_stats[g].gemm[0] + self._gpu_stats[g].ck_bwd[0] + self._gpu_stats[g].ck_fwd[0]
                + self._gpu_stats[g].te[0] + self._gpu_stats[g].memcpy[0] + self._gpu_stats[g].fillBuffer[0]
            )
            ov_ns, nonov_ns = rccl_overlap_two_pointer(comp, comm)
            self._gpu_stats[g].overlapped_comm_ns = ov_ns
            self._gpu_stats[g].non_overlapped_comm_ns = nonov_ns

        self._hip_events = build_hip_api_events(
            self.trace_events, min_tid=self.min_tid, max_tid=self.max_tid
        )
        self._hip_events = [e for e in self._hip_events if e.dur_ns >= self.min_event_ns]

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def used_fav3(self) -> bool:
        return self._used_fav3

    def get_df_category_summary(
        self,
        label_compute_fraction: str = "Fraction of compute runtime",
        label_total_fraction: str = "Fraction of total runtime",
    ) -> pd.DataFrame:
        return build_summary_dataframe(
            self._gpu_stats,
            label_compute_fraction=label_compute_fraction,
            label_total_fraction=label_total_fraction,
        )

    def get_xla_top(self, top_n: int = 30) -> List[Tuple[str, int, int, float]]:
        """Returns [(name, total_ns, count, fraction_of_xla), ...]."""
        xla_items = sorted(self._xla_agg.items(), key=lambda kv: kv[1][0], reverse=True)
        xla_total_ns = sum(ns for ns, _ in self._xla_agg.values()) or 1
        return [
            (name, ns, cnt, ns / xla_total_ns)
            for name, (ns, cnt) in xla_items[:top_n]
        ]

    def get_df_xla_top(self, top_n: int = 30) -> pd.DataFrame:
        rows = []
        xla_top = self.get_xla_top(top_n=top_n)
        xla_total_ns = sum(ns for ns, _ in self._xla_agg.values()) or 1
        for name, tot_ns, cnt, frac in xla_top:
            rows.append({
                "Kernel": name,
                "Total time (ms)": ns_to_ms(tot_ns),
                "Count": cnt,
                "Fraction of XLA": frac,
            })
        return pd.DataFrame(rows)

    def get_df_kernel_summary(self) -> pd.DataFrame:
        per_gpu_events: List[List[Event]] = []
        num_gpus = len(self._agents)
        for g in range(num_gpus):
            comp = [e for e in self._compute_events[g] if e.dur_ns >= self.min_event_ns]
            if self.kernel_summary_include_rccl:
                comp += [e for e in self._rccl_events[g] if e.dur_ns >= self.min_event_ns]
            per_gpu_events.append(comp)
        if self.kernel_summary_baseline == "total":
            baselines = [
                self._gpu_stats[g].compute_total_ns + self._gpu_stats[g].rccl[0]
                for g in range(num_gpus)
            ]
        else:
            baselines = [self._gpu_stats[g].compute_total_ns for g in range(num_gpus)]
        overall_baseline = sum(baselines)
        all_events = [e for sub in per_gpu_events for e in sub]
        if self.kernel_summary_group == "config":
            return build_kernel_summary_df_for_config(
                all_events, baseline_total_ns=overall_baseline, merge_names=self.merge_kernels
            )
        return build_kernel_summary_df_for_name(
            all_events, baseline_total_ns=overall_baseline, merge_names=self.merge_kernels
        )

    def get_df_hip_summary(self) -> pd.DataFrame:
        return build_hip_summary_df(self._hip_events, group=self.hip_summary_group)
