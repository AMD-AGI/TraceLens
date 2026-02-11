###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
HIP API ↔ Kernel correlation analysis for Perfetto-style traces (e.g. rocprofv3
exported as pftrace / traceEvents JSON). Builds summary tables of API launch
calls and linked kernel executions (T = API + Queue + Kernel time).
"""

import re
import statistics as stats
from collections import defaultdict
from typing import Any, Dict, List, Optional, Pattern, Tuple

import pandas as pd

import logging

logger = logging.getLogger(__name__)

# HIP launch API names and pattern
LAUNCH_HIP = {
    "hipModuleLaunchKernel",
    "hipLaunchKernelGGL",
    "hipLaunchKernel",
    "hipExtLaunchKernelGGL",
    "hipModuleLaunchCooperativeKernel",
    "hipExtModuleLaunchKernel",
    "hipMemset",
    "hipMemcpyHtoDAsync",
    "hipMemcpyDtoHAsync",
    "hipMemsetD32Async",
}
LAUNCH_REGEX = re.compile(r"(?:Launch).*Kernel", re.IGNORECASE)


def _extract_time_ns(e: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract begin_ns, end_ns, delta_ns from event (args or ts/dur in µs)."""
    args = e.get("args") or {}
    b, en, d = args.get("begin_ns"), args.get("end_ns"), args.get("delta_ns")
    if b is not None and en is not None and d is None:
        d = int(en) - int(b)
    if b is None or en is None or d is None:
        ts, dur = e.get("ts"), e.get("dur")
        if ts is not None and dur is not None:
            b, d = int(ts) * 1000, int(dur) * 1000
            en = b + d
        else:
            return None, None, None
    return int(b), int(en), int(d)


def _discover_agent_to_devid(events: List[Dict[str, Any]]) -> Dict[str, int]:
    agent_to_idx: Dict[str, int] = {}
    idx = 0
    for ev in events:
        agent = (ev.get("args") or {}).get("agent")
        if agent and agent not in agent_to_idx:
            agent_to_idx[agent] = idx
            idx += 1
    return agent_to_idx


def _get_device_id(e: Dict[str, Any], agent_to_idx: Dict[str, int]) -> Optional[int]:
    args = e.get("args") or {}
    for k in ("device_id", "deviceId", "dev_id", "DevId", "gpu_id", "gpuId"):
        if k in args:
            try:
                return int(args[k])
            except (TypeError, ValueError):
                pass
    agent = args.get("agent")
    if agent and agent in agent_to_idx:
        return agent_to_idx[agent]
    return None


def _is_hip_api_event(e: Dict[str, Any]) -> bool:
    if (e.get("cat") or "") != "hip_api":
        return False
    return (e.get("args") or {}).get("corr_id") is not None


def _is_launch_api(e: Dict[str, Any]) -> bool:
    if not _is_hip_api_event(e):
        return False
    name = (e.get("name") or "")
    return name in LAUNCH_HIP or bool(LAUNCH_REGEX.search(name))


def _is_kernel_event(e: Dict[str, Any]) -> bool:
    name = (e.get("name") or "").lower()
    cat = (e.get("cat") or "").lower()
    if name.endswith(".kd") or ".kd " in name:
        return True
    if "kernel_dispatch" in cat or "gpu_activity" in cat or "kernel" in cat or "memory_copy" in cat:
        _, _, d = _extract_time_ns(e)
        return d is not None and d > 0 and "register" not in name
    return False


def _get_corr_id(e: Dict[str, Any]) -> Optional[int]:
    args = e.get("args") or {}
    for k in ("corr_id", "correlation_id", "correlationId", "CorrId"):
        if k in args:
            try:
                return int(args[k])
            except (TypeError, ValueError):
                pass
    return None


def _get_kernel_name(e: Dict[str, Any]) -> str:
    args = e.get("args") or {}
    for k in ("kernel_name", "KernelName", "symbol", "symbol_name", "funcName"):
        if args.get(k):
            return str(args[k])
    return str(e.get("name") or "")


def _stats_vals(vals: List[int]) -> Tuple[float, float, int, int, float]:
    if not vals:
        return (0.0, 0.0, 0, 0, 0.0)
    arr = sorted(vals)
    n = len(arr)
    avg = sum(arr) / n
    med = float(arr[n // 2]) if n % 2 == 1 else (arr[n // 2 - 1] + arr[n // 2]) / 2.0
    mn, mx = arr[0], arr[-1]
    sd = stats.pstdev(arr) if n > 1 else 0.0
    return (avg, med, mn, mx, sd)


class PftraceHipApiAnalyzer:
    """
    Analyzer for Perfetto-style trace events: correlates HIP API (launch) events
    with kernel/dispatch events via correlation ID and produces summary DataFrames.
    """

    def __init__(
        self,
        events: List[Dict[str, Any]],
        exclude_kernel_re: Optional[Pattern[str]] = None,
        allow_multi_kernel_per_api: bool = False,
        include_nonlaunch_apis: bool = False,
    ):
        """
        Args:
            events: List of trace events (each a dict with cat, name, ts, dur, args, etc.).
            exclude_kernel_re: If set, kernel names matching this regex are excluded.
            allow_multi_kernel_per_api: If False, only the first kernel per correlation ID is used.
            include_nonlaunch_apis: If True, include API rows that have no linked kernel.
        """
        self.events = events
        self.exclude_kernel_re = exclude_kernel_re
        self.allow_multi_kernel_per_api = allow_multi_kernel_per_api
        self.include_nonlaunch_apis = include_nonlaunch_apis

    def get_df_api_kernel_summary(self) -> pd.DataFrame:
        """
        Build a summary DataFrame: one row per (PID, TID, DevId, API Name, Kernel Name)
        with count and statistics for T (total), A (API), Q (queue), K (kernel) in nanoseconds.

        Returns:
            DataFrame with columns: PID, TID, DevId, Count, QCount, TAvg_ns, TMed_ns, ...
            AAvg_ns, ..., QAvg_ns, ..., KAvg_ns, ..., API Name, Kernel Name.
        """
        agent_to_idx = _discover_agent_to_devid(self.events)
        api_by_corr: Dict[int, Dict[str, Any]] = {}
        kernels_by_corr: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for e in self.events:
            corr = _get_corr_id(e)
            if corr is None:
                continue
            if _is_kernel_event(e):
                kname = _get_kernel_name(e)
                if self.exclude_kernel_re and self.exclude_kernel_re.search(kname or ""):
                    continue
                kernels_by_corr[corr].append(e)
            elif _is_hip_api_event(e) or (e.get("cat") or "").lower() in ("cuda_api", "cupti"):
                api_by_corr[corr] = e

        packed_rows: List[Dict[str, Any]] = []
        for corr, api_e in api_by_corr.items():
            api_name = api_e.get("name") or ""
            if not self.include_nonlaunch_apis and not _is_launch_api(api_e):
                continue
            b, en, d = _extract_time_ns(api_e)
            if d is None:
                continue
            a_ns = int(d)
            api_end = en if en is not None else (b + d if b is not None else None)
            pid = int(api_e.get("pid", 0))
            tid = int(api_e.get("tid", 0))
            matched = kernels_by_corr.get(corr, [])
            if not matched:
                if self.include_nonlaunch_apis:
                    packed_rows.append(
                        dict(
                            pid=pid, tid=tid, dev=0, api=api_name, kern="",
                            count=1, qcount=0, T=[], A=[a_ns], Q=[0], K=[0],
                        )
                    )
                continue
            kernel_events = matched
            if not self.allow_multi_kernel_per_api and len(kernel_events) > 1:
                kernel_events = sorted(
                    kernel_events, key=lambda x: (_extract_time_ns(x)[0] or 0)
                )[:1]
            for ke in kernel_events:
                kb, _, kd = _extract_time_ns(ke)
                if kd is None or kb is None:
                    continue
                dev = _get_device_id(ke, agent_to_idx) or 0
                kname = _get_kernel_name(ke)
                q_ns = max(0, kb - api_end) if api_end is not None else 0
                k_ns = int(kd)
                t_ns = a_ns + q_ns + k_ns
                packed_rows.append(
                    dict(
                        pid=pid, tid=tid, dev=dev, api=api_name, kern=kname,
                        count=1, qcount=1, T=[t_ns], A=[a_ns], Q=[q_ns], K=[k_ns],
                    )
                )

        groups: Dict[Tuple[int, int, int, str, str], Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "qcount": 0, "T": [], "A": [], "Q": [], "K": []}
        )
        for r in packed_rows:
            key = (r["pid"], r["tid"], r["dev"], r["api"], r["kern"])
            g = groups[key]
            g["count"] += r["count"]
            g["qcount"] += r["qcount"]
            g["T"].extend(r["T"])
            g["A"].extend(r["A"])
            g["Q"].extend(r["Q"])
            g["K"].extend(r["K"])

        col_pid: List[int] = []
        col_tid: List[int] = []
        col_dev: List[int] = []
        col_count: List[int] = []
        col_qcount: List[int] = []
        col_t_avg: List[float] = []
        col_t_med: List[float] = []
        col_t_min: List[int] = []
        col_t_max: List[int] = []
        col_t_std: List[float] = []
        col_a_avg: List[float] = []
        col_a_med: List[float] = []
        col_a_min: List[int] = []
        col_a_max: List[int] = []
        col_a_std: List[float] = []
        col_q_avg: List[float] = []
        col_q_med: List[float] = []
        col_q_min: List[int] = []
        col_q_max: List[int] = []
        col_q_std: List[float] = []
        col_k_avg: List[float] = []
        col_k_med: List[float] = []
        col_k_min: List[int] = []
        col_k_max: List[int] = []
        col_k_std: List[float] = []
        col_api: List[str] = []
        col_kern: List[str] = []

        for key in sorted(groups.keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
            pid, tid, dev, api_name, kern_name = key
            g = groups[key]
            t_avg, t_med, t_min, t_max, t_std = _stats_vals(g["T"])
            a_avg, a_med, a_min, a_max, a_std = _stats_vals(g["A"])
            q_avg, q_med, q_min, q_max, q_std = _stats_vals(g["Q"])
            k_avg, k_med, k_min, k_max, k_std = _stats_vals(g["K"])
            col_pid.append(pid)
            col_tid.append(tid)
            col_dev.append(dev)
            col_count.append(g["count"])
            col_qcount.append(g["qcount"])
            col_t_avg.append(t_avg)
            col_t_med.append(t_med)
            col_t_min.append(int(t_min))
            col_t_max.append(int(t_max))
            col_t_std.append(t_std)
            col_a_avg.append(a_avg)
            col_a_med.append(a_med)
            col_a_min.append(int(a_min))
            col_a_max.append(int(a_max))
            col_a_std.append(a_std)
            col_q_avg.append(q_avg)
            col_q_med.append(q_med)
            col_q_min.append(int(q_min))
            col_q_max.append(int(q_max))
            col_q_std.append(q_std)
            col_k_avg.append(k_avg)
            col_k_med.append(k_med)
            col_k_min.append(int(k_min))
            col_k_max.append(int(k_max))
            col_k_std.append(k_std)
            col_api.append(api_name)
            col_kern.append(kern_name)

        return pd.DataFrame({
            "PID": col_pid,
            "TID": col_tid,
            "DevId": col_dev,
            "Count": col_count,
            "QCount": col_qcount,
            "TAvg_ns": col_t_avg,
            "TMed_ns": col_t_med,
            "TMin_ns": col_t_min,
            "TMax_ns": col_t_max,
            "TStdDev_ns": col_t_std,
            "AAvg_ns": col_a_avg,
            "AMed_ns": col_a_med,
            "AMin_ns": col_a_min,
            "AMax_ns": col_a_max,
            "AStdDev_ns": col_a_std,
            "QAvg_ns": col_q_avg,
            "QMed_ns": col_q_med,
            "QMin_ns": col_q_min,
            "QMax_ns": col_q_max,
            "QStdDev_ns": col_q_std,
            "KAvg_ns": col_k_avg,
            "KMed_ns": col_k_med,
            "KMin_ns": col_k_min,
            "KMax_ns": col_k_max,
            "KStdDev_ns": col_k_std,
            "API Name": col_api,
            "Kernel Name": col_kern,
        })
