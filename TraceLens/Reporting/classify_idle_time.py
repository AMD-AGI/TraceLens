#!/usr/bin/env python3
"""
Idle Time Classification for PyTorch Profiler Traces

Consumes a PyTorch trace, classifies each GPU idle interval along three axes
(noise/macro, drain_type, cpu_during_gap), and emits an augmented trace with
extra GPU-side annotation tracks visualizing the labels in Perfetto.

Usage:
    PYTHONPATH=/path/to/TraceLens python3 classify_idle_time.py <trace.json[.gz]> [--micro-thresh 5.0] [-o output.json.gz]
"""

import argparse
import bisect
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

from TraceLens import DataLoader, GPUEventAnalyser, TreePerfAnalyzer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNC_RUNTIME_NAMES = {
    "hipDeviceSynchronize", "cudaDeviceSynchronize",
    "hipStreamSynchronize", "cudaStreamSynchronize",
    "hipEventSynchronize", "cudaEventSynchronize",
}
MEMCPY_RUNTIME_NAMES = {
    "hipMemcpyDtoH", "cudaMemcpyDtoH",
    "hipMemcpyAsync", "cudaMemcpyAsync",
    "hipMemcpy", "cudaMemcpy",
    "hipMemcpyWithStream", "cudaMemcpyWithStream",
    "hipMemcpy2DAsync", "cudaMemcpy2DAsync",
}
ALLOC_FREE_NAMES = {
    "hipMalloc", "cudaMalloc", "hipFree", "cudaFree",
    "hipMallocAsync", "cudaMallocAsync", "hipFreeAsync", "cudaFreeAsync",
    "hipHostMalloc", "cudaHostAlloc", "hipHostFree", "cudaFreeHost",
}
LAUNCH_NAMES = {
    "hipLaunchKernel", "cudaLaunchKernel",
    "hipExtModuleLaunchKernel", "cuLaunchKernel",
    "hipGraphLaunch", "cudaGraphLaunch",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_sync_event(event, preceding_gpu_event=None):
    """Classify a CPU runtime event as a sync type, or return None.
    
    For memcpy runtime events, we check the preceding GPU event's kind field
    to determine D2H vs H2D, since the runtime event name alone (e.g.,
    hipMemcpyWithStream) doesn't tell us the direction.
    """
    name = event.get("name", "")
    if name in SYNC_RUNTIME_NAMES:
        return "DEVICE_SYNC" if "Device" in name else (
            "STREAM_SYNC" if "Stream" in name else "EVENT_SYNC"
        )
    if name in MEMCPY_RUNTIME_NAMES:
        # Determine direction from the kind field. The runtime event uses
        # numeric enums (1=HtoD, 2=DtoH, 3=DtoD) while the GPU-side event
        # uses strings ("HtoD", "DtoH", "DtoD").
        kind = event.get("args", {}).get("kind", "")
        if not kind and kind != 0 and preceding_gpu_event is not None:
            kind = preceding_gpu_event.get("args", {}).get("kind", "")
        kind_str = str(kind)
        if kind_str == "2" or "DtoH" in kind_str or "DtoH" in name:
            return "D2H_COPY"
        if kind_str == "1" or "HtoD" in kind_str or "HtoD" in name:
            return "H2D_COPY"
        return None
    return None


def is_memcpy_with_kind(event, kind_substring):
    """Check if a memcpy runtime event has a specific kind."""
    return kind_substring in event.get("args", {}).get("kind", "")


def classify_runtime_event(event):
    """Classify a runtime event into a sub-type for cause analysis."""
    name = event.get("name", "")
    if name in ALLOC_FREE_NAMES:
        return "MEMORY_ALLOC"
    if name in LAUNCH_NAMES:
        return "LAUNCH_STALL"
    if classify_sync_event(event) is not None:
        return "SYNC_CALL"
    return "OTHER_RUNTIME"


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def extract_idle_intervals(gpu_events):
    """
    Given a list of GPU event dicts (with 'ts' and 't_end'),
    merge them into a busy timeline and return idle gaps as (start, end) tuples.
    """
    intervals = [(e["ts"], e["t_end"]) for e in gpu_events]
    merged = GPUEventAnalyser.merge_intervals(intervals)
    if len(merged) < 2:
        return []
    idle = []
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_end > gap_start:
            idle.append((gap_start, gap_end))
    return idle


def build_sorted_cpu_events(tree):
    """Build a list of CPU runtime events sorted by ts for binary-search overlap queries."""
    runtime_events = []
    for event in tree.events:
        cat = event.get("cat", "")
        if cat in ("cuda_runtime", "cuda_driver") and event.get("ph") == "X":
            if "t_end" not in event:
                event["t_end"] = event["ts"] + event.get("dur", 0)
            runtime_events.append(event)
    runtime_events.sort(key=lambda e: e["ts"])
    return runtime_events


def build_sorted_cpu_ops(tree):
    """Build a list of cpu_op / python_function events sorted by ts."""
    ops = []
    for event in tree.events:
        cat = event.get("cat", "")
        if cat in ("cpu_op", "python_function") and event.get("ph") == "X":
            if "t_end" not in event:
                event["t_end"] = event["ts"] + event.get("dur", 0)
            ops.append(event)
    ops.sort(key=lambda e: e["ts"])
    return ops


def compute_self_times(overlapping_events, gap_start, gap_end):
    """Compute self-time (exclusive time) for each event in the gap.

    For each event clipped to [gap_start, gap_end], subtracts the union of
    all contained children's clipped intervals. Returns dict mapping
    event_name -> total self-time (summed across instances with that name).

    Containment uses original (pre-clip) boundaries so that a parent event
    spanning the entire gap correctly recognizes its child even when both
    clip to the same range.
    """
    events = sorted(overlapping_events, key=lambda e: (e["ts"], -e.get("dur", 0)))

    intervals = []
    for evt in events:
        orig_s = evt["ts"]
        orig_e = evt.get("t_end", orig_s + evt.get("dur", 0))
        cs = max(orig_s, gap_start)
        ce = min(orig_e, gap_end)
        if ce <= cs:
            continue
        intervals.append({
            "name": evt.get("name", "?"),
            "orig_s": orig_s,
            "orig_e": orig_e,
            "cs": cs,
            "ce": ce,
            "clipped_dur": ce - cs,
            "children_time": 0.0,
        })

    # O(n^2) -- n is typically < 50 events per gap.
    # Use original boundaries for containment to handle events that both
    # span the entire gap (clipped ranges would be identical).
    for i in range(len(intervals)):
        child_intervals = []
        for j in range(len(intervals)):
            if i == j:
                continue
            # j is a child of i if i's original span contains j's original span
            # (strict on at least one side to avoid mutual containment of identical events)
            i_contains_j = (
                intervals[i]["orig_s"] <= intervals[j]["orig_s"] and
                intervals[i]["orig_e"] >= intervals[j]["orig_e"] and
                (intervals[i]["orig_s"] < intervals[j]["orig_s"] or
                 intervals[i]["orig_e"] > intervals[j]["orig_e"])
            )
            if i_contains_j:
                child_intervals.append((intervals[j]["cs"], intervals[j]["ce"]))

        if child_intervals:
            child_intervals.sort()
            merged = [child_intervals[0]]
            for s, e in child_intervals[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            intervals[i]["children_time"] = sum(e - s for s, e in merged)

    self_times = defaultdict(float)
    for iv in intervals:
        self_t = iv["clipped_dur"] - iv["children_time"]
        if self_t > 0:
            self_times[iv["name"]] += self_t

    return dict(self_times)


class OverlapIndex:
    """Pre-built index for fast interval overlap queries on sorted events.

    Uses numpy arrays + binary search to avoid scanning irrelevant events.
    Query cost: O(log n + k) where k is the subset size after binary search
    narrowing (vectorized numpy comparison).
    """

    def __init__(self, sorted_events):
        import numpy as np
        self._events = sorted_events
        if sorted_events:
            self._ts = np.array([e["ts"] for e in sorted_events], dtype=np.float64)
            self._tend = np.array([e["t_end"] for e in sorted_events], dtype=np.float64)
        else:
            self._ts = np.empty(0, dtype=np.float64)
            self._tend = np.empty(0, dtype=np.float64)

    def query(self, interval_start, interval_end):
        """Return events overlapping [interval_start, interval_end]."""
        import numpy as np
        if len(self._events) == 0:
            return []
        right = int(np.searchsorted(self._ts, interval_end, side="left"))
        if right == 0:
            return []
        mask = self._tend[:right] > interval_start
        indices = np.nonzero(mask)[0]
        return [self._events[int(i)] for i in indices]


def get_overlapping_events(sorted_events_or_index, interval_start, interval_end):
    """Return events from a sorted list that overlap with [interval_start, interval_end].

    Accepts either a plain list (for backward compat / tests) or an OverlapIndex.
    """
    if isinstance(sorted_events_or_index, OverlapIndex):
        return sorted_events_or_index.query(interval_start, interval_end)
    sorted_events = sorted_events_or_index
    idx = bisect.bisect_left([e["ts"] for e in sorted_events], interval_start)
    start_idx = max(0, idx - 200)
    result = []
    for i in range(start_idx, len(sorted_events)):
        e = sorted_events[i]
        if e["ts"] >= interval_end:
            break
        if e["t_end"] > interval_start:
            result.append(e)
    return result


def build_gpu_kernel_map(tree):
    """Build sorted list of GPU events (kernel, memcpy, memset) with parent info."""
    gpu_events = []
    for event in tree.events:
        cat = event.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset") and event.get("ph") == "X":
            if "t_end" not in event:
                event["t_end"] = event["ts"] + event.get("dur", 0)
            gpu_events.append(event)
    gpu_events.sort(key=lambda e: e["ts"])
    return gpu_events


def find_launch_for_kernel(tree, kernel_event):
    """Walk up the parent chain from a GPU kernel to find its runtime launch event."""
    parent_uid = kernel_event.get("parent")
    if parent_uid is None:
        return None
    parent = tree.events_by_uid.get(parent_uid)
    if parent is None:
        return None
    if parent.get("cat") in ("cuda_runtime", "cuda_driver"):
        return parent
    return None


def compute_median_launch_latency(tree, gpu_events_sorted):
    """Compute the median launch-to-execution latency across individually-launched kernels.
    
    Excludes graph launches (hipGraphLaunch/cudaGraphLaunch) since they batch many
    kernels and have fundamentally different latency characteristics.
    """
    latencies = []
    for kernel in gpu_events_sorted:
        launch = find_launch_for_kernel(tree, kernel)
        if launch is None:
            continue
        launch_name = launch.get("name", "")
        if "GraphLaunch" in launch_name or "graphLaunch" in launch_name:
            continue
        launch_end = launch["ts"] + launch.get("dur", 0)
        latency = kernel["ts"] - launch_end
        if 0 <= latency < 1000:  # cap at 1ms to exclude clear outliers
            latencies.append(latency)
    if not latencies:
        return 5.0
    latencies.sort()
    return latencies[len(latencies) // 2]


def classify_idle_intervals(tree, micro_thresh_us=5.0, anomaly_multiplier=10.0):  # anomaly_multiplier kept for API compat, unused
    """
    Main classification function. Returns a list of dicts, one per idle interval:
    {
        'start': float, 'end': float, 'duration': float,
        'label_noise': bool,
        'drain_type': str,  # 'starved' | 'sync_drain'
        'sync_type': str or None,
        'cpu_during_gap': str,  # LAUNCH_ANOMALY | LAUNCH_OVERHEAD_ONLY | RUNTIME_DOMINATED | CPU_DOMINATED
        'cpu_during_gap_detail': str or None,
        'dominant_op': str or None,
        'preceding_gpu_event': str or None,
        'following_gpu_event': str or None,
        'following_launch_name': str or None,
        'launch_to_exec_us': float or None,
        'dispatch_delay_us': float or None,
        'kernel_prequeued': bool or None,
    }
    """
    # Build GPU event analyzer to get gpu event lists
    analyser = GPUEventAnalyser(tree.events)
    gpu_event_lists = analyser.get_gpu_event_lists()
    all_gpu = gpu_event_lists[GPUEventAnalyser.all_gpu_key]

    if not all_gpu:
        print("No GPU events found in trace.")
        return []

    # Extract idle intervals
    idle_intervals = extract_idle_intervals(all_gpu)
    print(f"Found {len(idle_intervals)} idle intervals on GPU timeline.")

    # Build lookup structures (using OverlapIndex for fast queries)
    sorted_runtime = build_sorted_cpu_events(tree)
    sorted_cpu_ops = build_sorted_cpu_ops(tree)
    rt_index = OverlapIndex(sorted_runtime)
    ops_index = OverlapIndex(sorted_cpu_ops)
    gpu_sorted = build_gpu_kernel_map(tree)
    gpu_ts_list = [e["ts"] for e in gpu_sorted]

    TYPICAL_LAUNCH_LATENCY = 8.0  # µs — empirical baseline for individual kernel launches
    print(f"Launch latency baseline: {TYPICAL_LAUNCH_LATENCY:.1f} µs")

    # Build merged intervals to find bounding GPU events for each gap
    all_gpu_intervals = [(e["ts"], e["t_end"]) for e in all_gpu]
    merged_busy = GPUEventAnalyser.merge_intervals(all_gpu_intervals)
    # Map: for gap between merged_busy[i] and merged_busy[i+1],
    # preceding busy interval ends at merged_busy[i][1], next starts at merged_busy[i+1][0]
    # We need the actual GPU event at those boundaries for reporting.
    # Build a quick lookup: ts -> gpu event name (for the event ending at that ts)
    gpu_event_by_tend = {}
    gpu_event_by_tstart = {}
    for e in all_gpu:
        t_end = e.get("t_end", e["ts"] + e.get("dur", 0))
        gpu_event_by_tend[round(t_end, 3)] = e.get("name", "unknown")
        gpu_event_by_tstart[round(e["ts"], 3)] = e.get("name", "unknown")

    def _find_gpu_name_near(lookup, target, tolerance=2.0):
        rounded = round(target, 3)
        if rounded in lookup:
            return lookup[rounded]
        for offset in [0.001, -0.001, 0.01, -0.01, 0.1, -0.1, 1.0, -1.0]:
            k = round(target + offset, 3)
            if k in lookup:
                return lookup[k]
        return None

    results = []
    for (gap_start, gap_end) in idle_intervals:
        duration = gap_end - gap_start
        rec = {
            "start": gap_start,
            "end": gap_end,
            "duration": duration,
            "label_noise": False,
            "drain_type": "starved",
            "sync_type": None,
            "sync_event_name": None,
            "sync_event_correlation": None,
            "sync_event_dur": None,
            "cpu_during_gap": "CPU_DOMINATED",
            "cpu_during_gap_detail": None,
            "dominant_op": None,
            "preceding_gpu_event": _find_gpu_name_near(gpu_event_by_tend, gap_start),
            "following_gpu_event": _find_gpu_name_near(gpu_event_by_tstart, gap_end),
            "following_launch_name": None,
            "following_gpu_uid": None,
            "following_launch_uid": None,
            "sync_event_uid": None,
            "launch_to_exec_us": None,
            "kernel_prequeued": None,
        }

        # --- Pass 0: Noise filter ---
        if duration < micro_thresh_us:
            rec["label_noise"] = True
            results.append(rec)
            continue

        # --- Find preceding GPU event and following kernel's launch ---
        # We need both before the drain type check, since a sync is only
        # causal if it ended before the following kernel's launch.
        idx_prec = bisect.bisect_left(gpu_ts_list, gap_start) - 1
        preceding_gpu_evt = None
        if 0 <= idx_prec < len(gpu_sorted):
            candidate = gpu_sorted[idx_prec]
            if abs(candidate.get("t_end", candidate["ts"] + candidate.get("dur", 0)) - gap_start) < 2.0:
                preceding_gpu_evt = candidate

        idx = bisect.bisect_left(gpu_ts_list, gap_end - 0.001)
        following_kernel = None
        for i in range(max(0, idx - 1), min(len(gpu_sorted), idx + 3)):
            if abs(gpu_sorted[i]["ts"] - gap_end) < 1.0:
                following_kernel = gpu_sorted[i]
                break

        if following_kernel is not None:
            rec["following_gpu_uid"] = following_kernel.get("UID")

        launch = None
        launch_end = None
        if following_kernel is not None:
            launch = find_launch_for_kernel(tree, following_kernel)
            if launch is not None:
                rec["following_launch_name"] = launch.get("name", "unknown")
                rec["following_launch_uid"] = launch.get("UID")
                launch_end = launch["ts"] + launch.get("dur", 0)
                rec["launch_to_exec_us"] = following_kernel["ts"] - launch_end
                prequeued = (launch_end + TYPICAL_LAUNCH_LATENCY) < gap_start
                rec["kernel_prequeued"] = prequeued

        # --- Pass 1: Drain type check ---
        # A sync is causal only if it ended before the launch of the succeeding
        # GPU event — that proves the CPU was blocked on the sync and could not
        # launch the next kernel until the sync returned.
        window_before_start = gap_start - 2000
        nearby_runtime = get_overlapping_events(rt_index, window_before_start, gap_start + 1)
        for rt_evt in reversed(nearby_runtime):
            rt_end = rt_evt["ts"] + rt_evt.get("dur", 0)
            if rt_end >= gap_start - 2 and rt_evt["ts"] < gap_start:
                sync_type = classify_sync_event(rt_evt, preceding_gpu_evt)
                if sync_type is not None:
                    # Verify the sync ended before the following kernel's launch
                    if launch is not None and rt_end > launch["ts"]:
                        continue  # sync still running when launch happened — not causal
                    rec["sync_type"] = sync_type
                    rec["sync_event_name"] = rt_evt.get("name")
                    rec["sync_event_uid"] = rt_evt.get("UID")
                    rec["sync_event_correlation"] = rt_evt.get("args", {}).get("External id",
                        rt_evt.get("args", {}).get("correlation"))
                    rec["sync_event_dur"] = rt_evt.get("dur")
                    # Did this sync actually drain the queue, or was GPU already idle?
                    DRAIN_OVERLAP_THRESH = 5.0  # µs
                    drains = False
                    if preceding_gpu_evt is not None:
                        gpu_end = preceding_gpu_evt.get("t_end",
                            preceding_gpu_evt["ts"] + preceding_gpu_evt.get("dur", 0))
                        overlap = gpu_end - rt_evt["ts"]
                        drains = overlap > DRAIN_OVERLAP_THRESH
                    rec["drain_type"] = "sync_drain" if drains else "starved"
                    break

        # --- Pass 2: CPU during gap classification ---
        LAUNCH_ANOMALY_THRESH_NONPREQUEUED = 10.0   # µs — typical launch-to-exec is 7-8µs
        LAUNCH_ANOMALY_THRESH_PREQUEUED = 5.0        # µs — gap should be ~0 if kernel was already queued
        launch_to_exec = rec.get("launch_to_exec_us")
        prequeued = rec.get("kernel_prequeued")

        if prequeued is not None and launch_to_exec is not None:
            if not prequeued and launch_to_exec > LAUNCH_ANOMALY_THRESH_NONPREQUEUED:
                rec["cpu_during_gap"] = "LAUNCH_ANOMALY"
                rec["cpu_during_gap_detail"] = f"launch_to_exec={launch_to_exec:.1f}µs (launched_during_gap)"
                results.append(rec)
                continue
            if prequeued and duration > LAUNCH_ANOMALY_THRESH_PREQUEUED:
                rec["cpu_during_gap"] = "LAUNCH_ANOMALY"
                rec["cpu_during_gap_detail"] = f"gap={duration:.1f}µs, launch_to_exec={launch_to_exec:.1f}µs (prequeued)"
                results.append(rec)
                continue

        APP_OVERHEAD_THRESH = max(2.0, 0.5 * TYPICAL_LAUNCH_LATENCY)
        if launch_to_exec is not None:
            app_overhead = duration - launch_to_exec
            if app_overhead < APP_OVERHEAD_THRESH:
                rec["cpu_during_gap"] = "LAUNCH_OVERHEAD_ONLY"
                rec["cpu_during_gap_detail"] = f"app_overhead={app_overhead:.1f}µs, launch_to_exec={launch_to_exec:.1f}µs"
                results.append(rec)
                continue

        # cpu_op events are hierarchical (parents contain children) so summing
        # their durations double-counts.  Runtime events are leaf-level and
        # non-overlapping on a single thread, so we check whether any single
        # runtime event occupies a significant fraction of the idle gap.
        # If the dominant runtime event covers ≥25% of the gap, we call it
        # RUNTIME_DOMINATED; otherwise CPU_DOMINATED.

        overlapping_rt = get_overlapping_events(rt_index, gap_start, gap_end)
        overlapping_ops = get_overlapping_events(ops_index, gap_start, gap_end)

        def clipped_dur(evt, gs, ge):
            s = max(evt["ts"], gs)
            e = min(evt.get("t_end", evt["ts"] + evt.get("dur", 0)), ge)
            return max(0, e - s)

        RUNTIME_FRACTION_THRESH = 0.25

        type_times = defaultdict(float)
        name_times = defaultdict(float)
        for rt_evt in overlapping_rt:
            rtype = classify_runtime_event(rt_evt)
            cd = clipped_dur(rt_evt, gap_start, gap_end)
            type_times[rtype] += cd
            name_times[rt_evt.get("name", "unknown")] += cd

        dominant_rt = max(type_times, key=type_times.get) if type_times else None
        dominant_rt_time = type_times.get(dominant_rt, 0) if dominant_rt else 0

        if dominant_rt and dominant_rt_time >= duration * RUNTIME_FRACTION_THRESH:
            dominant_rt_name = max(name_times, key=name_times.get) if name_times else ""
            rec["cpu_during_gap"] = "RUNTIME_DOMINATED"
            rec["cpu_during_gap_detail"] = f"{dominant_rt}: {dominant_rt_name}"
        else:
            CPU_UNTRACED_THRESH = 0.20
            self_times = compute_self_times(overlapping_ops, gap_start, gap_end)
            if self_times:
                total_self_time = sum(self_times.values())
                coverage = total_self_time / duration if duration > 0 else 0
                if coverage < CPU_UNTRACED_THRESH:
                    rec["cpu_during_gap"] = "CPU_UNTRACED"
                    rec["dominant_op"] = max(self_times, key=self_times.get)
                    rec["cpu_during_gap_detail"] = f"self_time_coverage={coverage:.0%} of gap"
                else:
                    rec["cpu_during_gap"] = "CPU_DOMINATED"
                    rec["dominant_op"] = max(self_times, key=self_times.get)
                    top_ops = sorted(self_times.items(), key=lambda x: -x[1])[:3]
                    detail_parts = [f"{name}: {t/duration*100:.0f}%" for name, t in top_ops]
                    rec["cpu_during_gap_detail"] = ", ".join(detail_parts)
            else:
                rec["cpu_during_gap"] = "CPU_UNTRACED"
                rec["dominant_op"] = "(no_cpu_op_overlap)"
                rec["cpu_during_gap_detail"] = "no cpu_op/python_function events overlap this gap"

        results.append(rec)

    return results


# ---------------------------------------------------------------------------
# Workload type detection
# ---------------------------------------------------------------------------

def detect_workload_tags(tree):
    """Detect workload characteristics from trace events.

    Returns a set of string tags, e.g. {'training', 'fsdp', 'compiled'}.
    Multiple tags can apply simultaneously.
    """
    from collections import Counter

    name_counts = Counter()
    cat_counts = Counter()
    annot_names = Counter()

    for e in tree.events:
        ph = e.get("ph")
        if ph not in ("X", "B", "E", "i"):
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        name_counts[name] += 1
        cat_counts[cat] += 1
        if cat == "user_annotation":
            annot_names[name] += 1

    def count_matching(substring, case_sensitive=False):
        total = 0
        for k, v in name_counts.items():
            if case_sensitive:
                if substring in k:
                    total += v
            else:
                if substring.lower() in k.lower():
                    total += v
        return total

    tags = set()

    # --- Training vs Inference ---
    n_backward = count_matching("backward") + count_matching("Backward", case_sensitive=True)
    n_autograd = count_matching("autograd")
    n_optimizer = count_matching("optimizer") + count_matching("adam") + count_matching("sgd")
    n_loss = count_matching("loss")

    is_training = n_backward > 50 or (n_autograd > 20 and (n_optimizer > 0 or n_loss > 5))
    if is_training:
        tags.add("training")
    else:
        tags.add("inference")

    # --- Distributed ---
    n_allreduce = count_matching("allreduce") + count_matching("all_reduce")
    n_nccl = count_matching("nccl") + count_matching("rccl")
    n_ddp = count_matching("DistributedDataParallel") + count_matching("ddp")
    n_fsdp = count_matching("FullyShardedDataParallel") + count_matching("fsdp")

    if n_fsdp > 0:
        tags.add("fsdp")
    if n_ddp > 0 or (n_allreduce > 5 and n_nccl > 5 and n_fsdp == 0):
        tags.add("ddp")
    if n_nccl > 5 or n_allreduce > 0:
        tags.add("distributed")

    # --- Compiled / Dynamo ---
    n_compiled = count_matching("CompiledFunction", case_sensitive=True) + count_matching("dynamo")
    if n_compiled > 5:
        tags.add("compiled")

    # --- Graph replay ---
    n_graph_launch = count_matching("GraphLaunch", case_sensitive=True) + count_matching("graphLaunch", case_sensitive=True)
    if n_graph_launch > 5:
        tags.add("graph_replay")

    # --- vLLM inference ---
    n_prefill = count_matching("prefill")
    n_decode = count_matching("decode")
    n_sampling = count_matching("sampl")
    if n_prefill > 0 and n_decode > 0:
        tags.add("vllm_inference")

    return tags


# ---------------------------------------------------------------------------
# Augmented trace generation
# ---------------------------------------------------------------------------

def assign_idle_ids(classified):
    """Assign stable idle_id: macro intervals get 0-indexed positive IDs, noise gets negative."""
    macro_id = 0
    noise_id = -1
    for rec in classified:
        if rec["label_noise"]:
            rec["idle_id"] = noise_id
            noise_id -= 1
        else:
            rec["idle_id"] = macro_id
            macro_id += 1


def make_annotation_events(classified, gpu_pid):
    """
    Generate Chrome trace events for three extra annotation tracks on the GPU process,
    one per classification axis.
    """
    # Use high tid numbers to avoid collision with real GPU streams
    TID_NOISE_MACRO = 9000
    TID_DRAIN = 9001
    TID_CPU_GAP = 9002

    annotation_events = []

    # Thread name metadata
    annotation_events.append({
        "ph": "M", "name": "thread_name",
        "pid": gpu_pid, "tid": TID_NOISE_MACRO,
        "args": {"name": "Idle: Noise/Macro"}
    })
    annotation_events.append({
        "ph": "M", "name": "thread_sort_index",
        "pid": gpu_pid, "tid": TID_NOISE_MACRO,
        "args": {"sort_index": TID_NOISE_MACRO}
    })
    annotation_events.append({
        "ph": "M", "name": "thread_name",
        "pid": gpu_pid, "tid": TID_DRAIN,
        "args": {"name": "Idle: Drain Type"}
    })
    annotation_events.append({
        "ph": "M", "name": "thread_sort_index",
        "pid": gpu_pid, "tid": TID_DRAIN,
        "args": {"sort_index": TID_DRAIN}
    })
    annotation_events.append({
        "ph": "M", "name": "thread_name",
        "pid": gpu_pid, "tid": TID_CPU_GAP,
        "args": {"name": "Idle: CPU During Gap"}
    })
    annotation_events.append({
        "ph": "M", "name": "thread_sort_index",
        "pid": gpu_pid, "tid": TID_CPU_GAP,
        "args": {"sort_index": TID_CPU_GAP}
    })

    for rec in classified:
        ts = rec["start"]
        dur = rec["duration"]
        idle_id = rec["idle_id"]

        # Track 1: Noise vs Macro
        if rec["label_noise"]:
            label = "noise"
        else:
            label = f"idle#{idle_id}"
        annotation_events.append({
            "ph": "X", "cat": "idle_classification",
            "name": label,
            "pid": gpu_pid, "tid": TID_NOISE_MACRO,
            "ts": ts, "dur": dur,
            "args": {"idle_id": idle_id, "duration_us": f"{dur:.2f}"},
        })

        if rec["label_noise"]:
            continue

        # Track 2: Drain type
        drain = rec["drain_type"]
        if drain == "sync_drain":
            drain_label = f"idle#{idle_id} sync_drain: {rec['sync_type']}"
        else:
            drain_label = f"idle#{idle_id} starved"
        annotation_events.append({
            "ph": "X", "cat": "idle_classification",
            "name": drain_label,
            "pid": gpu_pid, "tid": TID_DRAIN,
            "ts": ts, "dur": dur,
            "args": {
                "idle_id": idle_id,
                "drain_type": drain,
                "sync_type": rec["sync_type"] or "none",
            },
        })

        # Track 3: CPU during gap
        cpu_label = rec["cpu_during_gap"]
        if rec["cpu_during_gap_detail"]:
            cpu_label = f"{rec['cpu_during_gap']}: {rec['cpu_during_gap_detail']}"
        cpu_label = f"idle#{idle_id} {cpu_label}"
        annotation_events.append({
            "ph": "X", "cat": "idle_classification",
            "name": cpu_label,
            "pid": gpu_pid, "tid": TID_CPU_GAP,
            "ts": ts, "dur": dur,
            "args": {
                "idle_id": idle_id,
                "cpu_during_gap": rec["cpu_during_gap"],
                "detail": rec["cpu_during_gap_detail"] or "",
            },
        })

    return annotation_events


def find_gpu_pid(events):
    """Find the GPU process ID that has the most kernel/memcpy/memset events."""
    from collections import Counter
    kernel_pids = Counter(
        e["pid"] for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
    )
    if kernel_pids:
        return kernel_pids.most_common(1)[0][0]
    # Fallback: first GPU metadata
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_labels":
            if "GPU" in e.get("args", {}).get("labels", ""):
                return e["pid"]
    return 0


# ---------------------------------------------------------------------------
# Excel report generation
# ---------------------------------------------------------------------------

def generate_excel_report(classified, output_path):
    """Generate an Excel workbook with idle_summary and idle_intervals sheets."""
    import numpy as np
    import pandas as pd

    macro = [r for r in classified if not r["label_noise"]]
    noise_count = sum(1 for r in classified if r["label_noise"])

    # --- Sheet 1: idle_summary ---
    # Group by (drain_type, cpu_during_gap, grouping_key) where grouping_key is
    # the dominant op for CPU_DOMINATED or the runtime sub-type for RUNTIME_DOMINATED.
    def get_grouping_key(rec):
        if rec["cpu_during_gap"] == "RUNTIME_DOMINATED":
            return rec["cpu_during_gap_detail"] or "OTHER_RUNTIME"
        if rec["cpu_during_gap"] == "CPU_DOMINATED":
            return rec["dominant_op"] or "unknown"
        if rec["cpu_during_gap"] == "CPU_UNTRACED":
            return rec["dominant_op"] or "(no_cpu_op_overlap)"
        if rec["cpu_during_gap"] == "LAUNCH_ANOMALY":
            return "prequeued" if rec.get("kernel_prequeued") else "launched_during_gap"
        if rec["cpu_during_gap"] == "LAUNCH_OVERHEAD_ONLY":
            return "prequeued" if rec.get("kernel_prequeued") else "launched_during_gap"
        return rec.get("cpu_during_gap_detail", "")

    def compute_stats(items):
        durations = [r["duration"] for r in items]
        arr = np.array(durations)
        return {
            "count": len(items),
            "total_time_ms": arr.sum() / 1e3,
            "mean_us": arr.mean(),
            "median_us": float(np.median(arr)),
            "std_us": arr.std(),
            "min_us": arr.min(),
            "max_us": arr.max(),
        }

    summary_rows = []

    # Total row
    if macro:
        total_stats = compute_stats(macro)
        total_stats["drain_type"] = "ALL"
        total_stats["cpu_during_gap"] = "ALL"
        total_stats["dominant_op"] = "ALL"
        total_stats["pct_of_idle"] = 100.0
        summary_rows.append(total_stats)

    # Noise row — with actual stats
    noise_intervals = [r for r in classified if r["label_noise"]]
    if noise_intervals:
        noise_stats = compute_stats(noise_intervals)
        noise_stats["drain_type"] = "—"
        noise_stats["cpu_during_gap"] = "noise"
        noise_stats["dominant_op"] = "—"
        noise_stats["pct_of_idle"] = None
        summary_rows.append(noise_stats)
    else:
        summary_rows.append({
            "drain_type": "—", "cpu_during_gap": "noise", "dominant_op": "—",
            "count": 0, "total_time_ms": 0.0,
            "pct_of_idle": None, "mean_us": None, "median_us": None,
            "std_us": None, "min_us": None, "max_us": None,
        })

    # Group by combo
    from itertools import groupby as itertools_groupby
    total_macro_time = sum(r["duration"] for r in macro) if macro else 1.0

    combo_groups = defaultdict(list)
    for rec in macro:
        key = (rec["drain_type"], rec["cpu_during_gap"], get_grouping_key(rec))
        combo_groups[key].append(rec)

    # Sort by total time descending
    sorted_combos = sorted(combo_groups.items(), key=lambda x: -sum(r["duration"] for r in x[1]))
    for (drain, cpu_gap, dom_op), items in sorted_combos:
        stats = compute_stats(items)
        stats["drain_type"] = drain
        stats["cpu_during_gap"] = cpu_gap
        stats["dominant_op"] = dom_op
        stats["pct_of_idle"] = stats["total_time_ms"] / (total_macro_time / 1e3) * 100
        ids = sorted(r.get("idle_id", -1) for r in items)
        stats["idle_ids"] = ",".join(str(x) for x in ids)
        summary_rows.append(stats)

    summary_col_order = [
        "drain_type", "cpu_during_gap", "dominant_op", "count",
        "total_time_ms", "pct_of_idle", "cumulative_pct", "mean_us", "median_us",
        "std_us", "min_us", "max_us", "idle_ids",
    ]
    df_summary = pd.DataFrame(summary_rows, columns=summary_col_order)

    # Cumulative % only for the combo rows (skip ALL and noise rows)
    combo_mask = ~df_summary["cpu_during_gap"].isin(["ALL", "noise"])
    df_summary.loc[combo_mask, "cumulative_pct"] = (
        df_summary.loc[combo_mask, "pct_of_idle"].cumsum()
    )

    # --- Sheet 2: idle_intervals ---
    interval_rows = []
    for i, rec in enumerate(macro):
        group = f"{rec['drain_type']} | {rec['cpu_during_gap']} | {get_grouping_key(rec)}"
        interval_rows.append({
            "idle_id": rec.get("idle_id", i),
            "group": group,
            "start_us": rec["start"],
            "end_us": rec["end"],
            "duration_us": rec["duration"],
            "drain_type": rec["drain_type"],
            "sync_type": rec["sync_type"],
            "sync_event_name": rec.get("sync_event_name"),
            "sync_event_correlation": rec.get("sync_event_correlation"),
            "sync_event_dur": rec.get("sync_event_dur"),
            "cpu_during_gap": rec["cpu_during_gap"],
            "cpu_during_gap_detail": rec["cpu_during_gap_detail"],
            "dominant_op": rec["dominant_op"],
            "preceding_gpu_event": rec.get("preceding_gpu_event"),
            "following_gpu_event": rec.get("following_gpu_event"),
            "following_launch_name": rec.get("following_launch_name"),
            "following_gpu_uid": rec.get("following_gpu_uid"),
            "following_launch_uid": rec.get("following_launch_uid"),
            "sync_event_uid": rec.get("sync_event_uid"),
            "launch_to_exec_us": rec.get("launch_to_exec_us"),
            "kernel_prequeued": rec.get("kernel_prequeued"),
        })

    df_intervals = pd.DataFrame(interval_rows)
    if not df_intervals.empty:
        df_intervals = df_intervals.sort_values("duration_us", ascending=False).reset_index(drop=True)

    # --- Sheet 0: idle_overview (coarse summary) ---
    overview_rows = []
    if macro:
        total_stats = compute_stats(macro)
        total_stats.update({"drain_type": "ALL", "cpu_during_gap": "ALL", "pct_of_idle": 100.0})
        overview_rows.append(total_stats)

    coarse_groups = defaultdict(list)
    for rec in macro:
        coarse_groups[(rec["drain_type"], rec["cpu_during_gap"])].append(rec)

    sorted_coarse = sorted(coarse_groups.items(), key=lambda x: -sum(r["duration"] for r in x[1]))
    for (drain, cpu_gap), items in sorted_coarse:
        stats = compute_stats(items)
        stats["drain_type"] = drain
        stats["cpu_during_gap"] = cpu_gap
        stats["pct_of_idle"] = stats["total_time_ms"] / (total_macro_time / 1e3) * 100
        overview_rows.append(stats)

    overview_col_order = [
        "drain_type", "cpu_during_gap", "count",
        "total_time_ms", "pct_of_idle", "mean_us", "median_us",
        "min_us", "max_us",
    ]
    df_overview = pd.DataFrame(overview_rows, columns=overview_col_order)

    # Write Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_overview.to_excel(writer, sheet_name="idle_overview", index=False)
        df_summary.to_excel(writer, sheet_name="idle_summary", index=False)
        df_intervals.to_excel(writer, sheet_name="idle_intervals", index=False)

    print(f"Wrote Excel report: {output_path}")
    print(f"  idle_overview: {len(df_overview)} rows")
    print(f"  idle_summary: {len(df_summary)} rows")
    print(f"  idle_intervals: {len(df_intervals)} rows")


def main():
    parser = argparse.ArgumentParser(description="Classify GPU idle time intervals in a PyTorch trace")
    parser.add_argument("trace", help="Path to PyTorch trace JSON (or .json.gz)")
    parser.add_argument("--micro-thresh", type=float, default=5.0,
                        help="Micro idle threshold in µs (default: 5.0)")
    parser.add_argument("--anomaly-multiplier", type=float, default=10.0,
                        help="Multiplier on median launch latency for anomaly detection (default: 10.0)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for augmented trace (default: <input>_idle_classified.json.gz)")
    args = parser.parse_args()

    trace_path = args.trace
    if args.output is None:
        p = Path(trace_path)
        # Strip all suffixes like .pt.trace.json.gz
        stem = p.name
        for _ in range(4):
            if "." in stem:
                stem = stem.rsplit(".", 1)[0]
        output_path = str(p.parent / f"{stem}_idle_classified.json.gz")
    else:
        output_path = args.output

    # --- Load trace via TraceLens ---
    print(f"Loading trace: {trace_path}")
    perf_analyzer = TreePerfAnalyzer.from_file(trace_path)
    tree = perf_analyzer.tree

    # --- Classify idle intervals ---
    print("Classifying idle intervals...")
    classified = classify_idle_intervals(
        tree,
        micro_thresh_us=args.micro_thresh,
        anomaly_multiplier=args.anomaly_multiplier,
    )

    # --- Print summary ---
    noise_count = sum(1 for r in classified if r["label_noise"])
    macro = [r for r in classified if not r["label_noise"]]
    macro_total_us = sum(r["duration"] for r in macro)

    print(f"\n{'='*60}")
    print(f"Idle Interval Summary")
    print(f"{'='*60}")
    print(f"Total intervals:    {len(classified)}")
    print(f"  Noise (<{args.micro_thresh}µs):  {noise_count}")
    print(f"  Macro (≥{args.micro_thresh}µs):  {len(macro)}  ({macro_total_us/1e3:.2f} ms total)")

    if macro:
        sync_drain = [r for r in macro if r["drain_type"] == "sync_drain"]
        starved = [r for r in macro if r["drain_type"] == "starved"]
        print(f"\nDrain Type:")
        print(f"  sync_drain:       {len(sync_drain)}  ({sum(r['duration'] for r in sync_drain)/1e3:.2f} ms)")
        print(f"  starved:          {len(starved)}  ({sum(r['duration'] for r in starved)/1e3:.2f} ms)")
        from collections import Counter
        if sync_drain:
            for st, cnt in Counter(r["sync_type"] for r in sync_drain).most_common():
                st_time = sum(r["duration"] for r in sync_drain if r["sync_type"] == st)
                print(f"    {st}: {cnt} intervals, {st_time/1e3:.2f} ms")

        print(f"\nCPU During Gap:")
        for cpu_gap in ["LAUNCH_ANOMALY", "LAUNCH_OVERHEAD_ONLY", "RUNTIME_DOMINATED", "CPU_DOMINATED"]:
            items = [r for r in macro if r["cpu_during_gap"] == cpu_gap]
            if items:
                total_t = sum(r["duration"] for r in items)
                print(f"  {cpu_gap}: {len(items)} intervals, {total_t/1e3:.2f} ms")
                if cpu_gap == "RUNTIME_DOMINATED":
                    for detail, cnt in Counter(r["cpu_during_gap_detail"] for r in items).most_common():
                        dt = sum(r["duration"] for r in items if r["cpu_during_gap_detail"] == detail)
                        print(f"    {detail}: {cnt} intervals, {dt/1e3:.2f} ms")

    # --- Assign stable idle IDs before any output ---
    assign_idle_ids(classified)

    # --- Generate Excel report ---
    excel_path = output_path.replace(".json.gz", ".xlsx").replace(".json", ".xlsx")
    if excel_path == output_path:
        excel_path = output_path + ".xlsx"
    generate_excel_report(classified, excel_path)

    # --- Load raw trace data and augment ---
    print(f"\nLoading raw trace for augmentation...")
    raw_data = DataLoader.load_data(trace_path)
    raw_events = raw_data["traceEvents"]
    gpu_pid = find_gpu_pid(raw_events)
    print(f"GPU pid: {gpu_pid}")

    annotation_events = make_annotation_events(classified, gpu_pid)
    raw_events.extend(annotation_events)
    raw_data["traceEvents"] = raw_events

    print(f"Added {len(annotation_events)} annotation events to trace.")
    print(f"Writing augmented trace to: {output_path}")
    with gzip.open(output_path, "wt") as f:
        json.dump(raw_data, f)

    print("Done.")


if __name__ == "__main__":
    main()
