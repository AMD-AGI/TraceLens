###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GPU idle gap classification: measure each gap's parts, then label by the largest part.

Every gap on the merged GPU timeline is split into parts that add up to the gap, using
only the thread that launched the GPU op ending the gap:

    unqueued (launch returned less than launch_latency_us before the gap):
        cpu_op_us, cpu_runtime_us, cpu_untraced_us, launch_call_us, submit_latency_us
    queued (launch returned at least launch_latency_us before the gap): gpu_wait_us
    no_launch (no traced launch for the next op): unattributed_us

Independent of the label, drain_type records whether a sync on the launching thread
drained the GPU queue (sync_drain) or the CPU simply did not submit work in time (starved),
and stream_switch whether the ops on either side of the gap ran on different streams.
"""

import bisect
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser
from TraceLens.util import merge_intervals

SYNC_RUNTIME_NAMES = {
    "hipDeviceSynchronize",
    "cudaDeviceSynchronize",
    "hipStreamSynchronize",
    "cudaStreamSynchronize",
    "hipEventSynchronize",
    "cudaEventSynchronize",
}
# Documented to perform an implicit device synchronize.
IMPLICIT_SYNC_NAMES = {
    "hipFree",
    "cudaFree",
    "hipHostFree",
    "hipFreeHost",
    "cudaFreeHost",
}
# Copies block the host in either direction when host memory is pageable, even the
# "Async" forms; whether a given call blocked is decided by timing, not by name.
MEMCPY_RUNTIME_NAMES = {
    "hipMemcpyDtoH",
    "cudaMemcpyDtoH",
    "hipMemcpyAsync",
    "cudaMemcpyAsync",
    "hipMemcpy",
    "cudaMemcpy",
    "hipMemcpyWithStream",
    "cudaMemcpyWithStream",
    "hipMemcpy2DAsync",
    "cudaMemcpy2DAsync",
}
SYNC_CALL_NAMES = SYNC_RUNTIME_NAMES | IMPLICIT_SYNC_NAMES | MEMCPY_RUNTIME_NAMES
ALLOC_FREE_NAMES = {
    "hipMalloc",
    "cudaMalloc",
    "hipFree",
    "cudaFree",
    "hipMallocAsync",
    "cudaMallocAsync",
    "hipFreeAsync",
    "cudaFreeAsync",
    "hipHostMalloc",
    "cudaHostAlloc",
    "hipHostFree",
    "cudaFreeHost",
}
LAUNCH_NAMES = {
    "hipLaunchKernel",
    "cudaLaunchKernel",
    "hipExtModuleLaunchKernel",
    "cuLaunchKernel",
    "cudaLaunchKernelExC",
    "hipGraphLaunch",
    "cudaGraphLaunch",
}

RUNTIME_CATS = ("cuda_runtime", "cuda_driver")
CPU_OP_CATS = ("cpu_op", "python_function", "user_annotation")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

PARTS = (
    "cpu_op_us",
    "cpu_runtime_us",
    "cpu_untraced_us",
    "launch_call_us",
    "submit_latency_us",
    "gpu_wait_us",
    "unattributed_us",
)
LABELS = (
    "CPU_OP_DOMINATED",
    "RUNTIME_DOMINATED",
    "UNTRACED_DOMINATED",
    "LAUNCH_DOMINATED",
    "UNEXPLAINED_GPU_WAIT",
    "UNATTRIBUTED",
)


@dataclass(frozen=True)
class IdleThresholds:
    """Thresholds for idle gap classification (all times in µs)."""

    # Gaps shorter than this are noise and not classified.
    noise_us: float = 5.0
    # Normal launch return -> op start. An op is queued if its launch returned more than
    # this before the gap; launch -> start above this means the GPU did not start a
    # submitted op (UNEXPLAINED_GPU_WAIT/unqueued).
    launch_latency_us: float = 15.0
    # If set, the launch latency limit becomes
    # max(launch_latency_us, launch_latency_rel * median launch -> start of ops launched
    # during gaps), so it adapts to the platform.
    launch_latency_rel: Optional[float] = None
    # Another thread inside a runtime call for more than this fraction of the gap labels it
    # RUNTIME_DOMINATED/other_thread.
    other_thread_frac: float = 0.5
    # A sync drains the queue only if it started this long before the GPU finished.
    drain_overlap_us: float = 5.0
    # Reference CPU duration of one launch call; launch_call_x_ref is the next op's
    # launch call duration over this. Kernel launches take a few µs, graph launches
    # normally take much longer.
    launch_call_ref_us: float = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _t_end(e):
    return e.get("t_end", e["ts"] + e.get("dur", 0))


def stream_of(gpu_event):
    args = gpu_event.get("args", {})
    return args.get("stream", args.get("stream id", gpu_event.get("tid")))


def is_sync_call(name):
    """True for runtime calls that can block the host until the GPU catches up."""
    return name in SYNC_CALL_NAMES


def classify_runtime_event(event):
    """Classify a runtime event into a sub-type for cause analysis."""
    name = event.get("name", "")
    if name in ALLOC_FREE_NAMES:
        return "MEMORY_ALLOC"
    if name in LAUNCH_NAMES:
        return "LAUNCH_STALL"
    if is_sync_call(name):
        return "SYNC_CALL"
    return "OTHER_RUNTIME"


def is_enqueue_call(name):
    """True for calls that only enqueue work (kernel launches, async copies/memsets)."""
    return name in LAUNCH_NAMES or "Launch" in name or name.endswith("Async")


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+|0x[0-9a-fA-F]{6,}")


def strip_addresses(name):
    """Drop memory addresses (``<... object at 0x7f06...>``) so names group across runs."""
    return _ADDRESS.sub("", name).strip() if name else name


def extract_idle_intervals(gpu_events):
    """Merge GPU events (with 'ts' and 't_end') into a busy timeline; return idle gaps."""
    merged = merge_intervals([(e["ts"], e["t_end"]) for e in gpu_events])
    return [
        (merged[i][1], merged[i + 1][0])
        for i in range(len(merged) - 1)
        if merged[i + 1][0] > merged[i][1]
    ]


def gpu_busy_time_us(gpu_events):
    """Union of GPU event intervals (µs)."""
    return sum(
        e - s for s, e in merge_intervals([(e["ts"], e["t_end"]) for e in gpu_events])
    )


def build_sorted_cpu_events(tree):
    """CPU runtime events sorted by ts, for overlap queries."""
    runtime_events = []
    for event in tree.events:
        if event.get("cat", "") in RUNTIME_CATS and event.get("ph") == "X":
            event.setdefault("t_end", event["ts"] + event.get("dur", 0))
            runtime_events.append(event)
    runtime_events.sort(key=lambda e: e["ts"])
    return runtime_events


class OverlapIndex:
    """Interval overlap queries on events sorted by ts: O(log n + k) per query."""

    def __init__(self, sorted_events):
        import numpy as np

        self._events = sorted_events
        self._ts = np.array([e["ts"] for e in sorted_events], dtype=np.float64)
        self._tend = np.array([e["t_end"] for e in sorted_events], dtype=np.float64)

    def query(self, interval_start, interval_end):
        """Return events overlapping [interval_start, interval_end)."""
        import numpy as np

        if len(self._events) == 0:
            return []
        right = int(np.searchsorted(self._ts, interval_end, side="left"))
        if right == 0:
            return []
        indices = np.nonzero(self._tend[:right] > interval_start)[0]
        return [self._events[int(i)] for i in indices]


def build_gpu_kernel_map(tree):
    """GPU events (kernel, memcpy, memset) sorted by ts."""
    gpu_events = []
    for event in tree.events:
        if event.get("cat", "") in GPU_CATS and event.get("ph") == "X":
            event.setdefault("t_end", event["ts"] + event.get("dur", 0))
            gpu_events.append(event)
    gpu_events.sort(key=lambda e: e["ts"])
    return gpu_events


def find_launch_for_kernel(tree, kernel_event):
    """Return the runtime event that launched a GPU event, or None."""
    parent = tree.events_by_uid.get(kernel_event.get("parent"))
    if parent is not None and parent.get("cat") in RUNTIME_CATS:
        return parent
    return None


def find_launch_parent_op(tree, launch):
    """Name of the nearest cpu_op enclosing a launch call (e.g. ``aten::addmm``)."""
    parent = tree.events_by_uid.get(launch.get("parent"))
    while parent is not None and parent.get("cat") != "cpu_op":
        parent = tree.events_by_uid.get(parent.get("parent"))
    return parent.get("name") if parent is not None else None


def build_thread_index(tree):
    """(pid, tid) -> (events sorted outer-first, start times, parent index) for CPU events."""
    by_thread = defaultdict(list)
    for e in tree.events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if cat in RUNTIME_CATS or cat in CPU_OP_CATS:
            by_thread[(e.get("pid"), e.get("tid"))].append(e)
    index = {}
    for key, evs in by_thread.items():
        evs.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
        parent, stack = [], []
        for i, e in enumerate(evs):
            while stack and _t_end(evs[stack[-1]]) <= e["ts"]:
                stack.pop()
            parent.append(stack[-1] if stack else -1)
            stack.append(i)
        index[key] = (evs, [e["ts"] for e in evs], parent)
    return index


def exclusive_split(
    thread_index, thread_key, w0, w1, exclude_uid=None, return_event=False
):
    """Split [w0, w1) on one thread by exclusive time.

    Returns (parts, names, covering): parts has cpu_op_us / cpu_runtime_us /
    cpu_untraced_us summing to w1 - w0; names maps (part, event name) -> exclusive µs;
    covering is the deepest CPU op (not a runtime call or annotation) spanning at least
    half the window, else the CPU op with the most time in it: its name, or the event
    itself if return_event.
    ``user_annotation`` ranges (ProfilerStep#N, record_function) wrap work but are not
    work, so their exclusive time counts as untraced.
    """
    parts = {"cpu_op_us": 0.0, "cpu_runtime_us": 0.0, "cpu_untraced_us": 0.0}
    names = defaultdict(float)
    if w1 <= w0:
        return parts, names, None
    window = _window_events(thread_index, thread_key, w0, w1)
    candidates, events = _clipped_candidates(window, w0, w1, exclude_uid)
    best = _sweep_exclusive(candidates, events, w0, w1, parts, names)
    if return_event or best is None:
        return parts, names, best
    return parts, names, best.get("name", "?")


def _window_events(thread_index, thread_key, w0, w1):
    """Events on one thread open at w0 or starting in [w0, w1)."""
    evs, starts, parent = thread_index.get(thread_key, ([], [], []))
    lo = bisect.bisect_left(starts, w0)
    hi = bisect.bisect_left(starts, w1)
    # Events open at w0 are on the ancestor chain of the last event starting before w0.
    enclosing, j = [], lo - 1
    while j >= 0:
        if _t_end(evs[j]) > w0:
            enclosing.append(evs[j])
        j = parent[j]
    return enclosing + evs[lo:hi]


def _part_of(event):
    cat = event.get("cat")
    if cat in RUNTIME_CATS:
        return "cpu_runtime_us"
    if cat == "user_annotation":
        return "cpu_untraced_us"
    return "cpu_op_us"


def _clipped_candidates(window, w0, w1, exclude_uid):
    """Events clipped to [w0, w1), sorted for the sweep, and the events they index."""
    candidates, events = [], []
    for e in window:
        if exclude_uid is not None and e.get("UID") == exclude_uid:
            continue
        s, en = max(e["ts"], w0), min(_t_end(e), w1)
        if en <= s:
            continue
        # Same clipped start: outer (later clipped end, then longer) first so children
        # stack on top of their parents.
        candidates.append(
            (
                s,
                -en,
                -(_t_end(e) - e["ts"]),
                _part_of(e),
                e.get("name", "?"),
                len(events),
            )
        )
        events.append(e)
    candidates.sort()
    return candidates, events


def _sweep_exclusive(candidates, events, w0, w1, parts, names):
    """Credit each instant of [w0, w1) to the innermost open event in parts and names.

    Returns the covering CPU op: the deepest spanning at least half the window, else the
    one with the most time in it, else None.
    """
    stack = []  # [end, part, name]
    cur = w0

    def credit(upto):
        nonlocal cur
        if upto <= cur:
            return
        if stack:
            parts[stack[-1][1]] += upto - cur
            names[(stack[-1][1], stack[-1][2])] += upto - cur
        else:
            parts["cpu_untraced_us"] += upto - cur
        cur = upto

    covering, largest, largest_t = None, None, 0.0
    for s, neg_en, _, part, name, i in candidates:
        en = -neg_en
        while stack and stack[-1][0] <= s:
            credit(stack[-1][0])
            stack.pop()
        credit(s)
        if stack:
            en = min(en, stack[-1][0])
        stack.append([en, part, name])
        if part == "cpu_op_us":
            if en - s >= 0.5 * (w1 - w0):
                covering = events[i]
            # >= so a nested op with the same overlap as its parent wins.
            if en - s >= largest_t:
                largest, largest_t = events[i], en - s
    while stack:
        credit(stack[-1][0])
        stack.pop()
    credit(w1)
    return covering or largest


def _top_name(names, part):
    best, best_t = None, 0.0
    for (p, name), t in names.items():
        if p == part and t > best_t:
            best, best_t = name, t
    return best


def _new_record(gap_start, gap_end):
    rec = {
        "start": gap_start,
        "end": gap_end,
        "duration": gap_end - gap_start,
        "label_noise": False,
        "case": None,
        "label": None,
        "sub_label": None,
        "detail": None,
        "drain_type": None,
        "stream_switch": None,
        "sync_event_name": None,
        "sync_event_correlation": None,
        "sync_event_dur": None,
        "sync_gain_us": None,
        "sync_event_uid": None,
        "sync_entry_point": None,
        "sync_call_stack_full": None,
        "covering_op": None,
        "covering_op_uid": None,
        "covering_call_stack_full": None,
        "top_op": None,
        "top_runtime": None,
        "following_launch_name": None,
        "launch_parent_op": None,
        "launch_tid": None,
        "launch_to_start_us": None,
        "launch_lead_us": None,
        "launch_call_x_ref": None,
        "other_thread_top_runtime": None,
        "other_thread_runtime_us": None,
        "preceding_gpu_event": None,
        "following_gpu_event": None,
        "preceding_stream": None,
        "following_stream": None,
        "preceding_gpu_uid": None,
        "following_gpu_uid": None,
        "following_launch_uid": None,
    }
    rec.update(dict.fromkeys(PARTS, 0.0))
    return rec


def _find_sync_drain(rec, rt_index, gap_start, launch, gpu_by_end, i, th):
    """Set drain_type and sync_* fields.

    A sync drains the queue if it is on the launching thread, overlaps the gap start,
    returned before the next launch began, and started more than drain_overlap_us before
    the GPU finished other work. A copy's own transfer is not queued work: a copy issued
    to an idle GPU that then blocks the host is starved, not a drain.
    sync_gain_us = min(sync duration, gap) bounds the idle that removing that sync could
    save; it counts only this sync, not later ones in the same gap.
    """
    rec["drain_type"] = "starved"
    thread = (launch.get("pid"), launch.get("tid")) if launch is not None else None
    for rt in sorted(
        rt_index.query(gap_start - 2.0, gap_start), key=lambda e: -e["ts"]
    ):
        if thread is not None and (rt.get("pid"), rt.get("tid")) != thread:
            continue
        if not is_sync_call(rt.get("name", "")):
            continue
        if launch is not None and _t_end(rt) > launch["ts"]:
            continue  # still running when the next launch happened: not causal
        rec["sync_event_name"] = rt.get("name")
        rec["sync_event_uid"] = rt.get("UID")
        args = rt.get("args", {})
        rec["sync_event_correlation"] = args.get("External id", args.get("correlation"))
        rec["sync_event_dur"] = rt.get("dur")
        k = i
        while k >= 0 and gpu_by_end[k].get("parent") == rt.get("UID"):
            k -= 1
        work = gpu_by_end[k] if k >= 0 else None
        if work is not None and _t_end(work) - rt["ts"] > th.drain_overlap_us:
            rec["drain_type"] = "sync_drain"
            rec["sync_gain_us"] = min(rt.get("dur", 0.0), rec["duration"])
        return


def _fmt_parts(rec):
    nonzero = [(p, rec[p]) for p in PARTS if rec[p] > 0.05]
    nonzero.sort(key=lambda x: -x[1])
    return ", ".join(f"{p[:-3]}={v:.1f}us" for p, v in nonzero)


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------


def call_stack_full(tree, event):
    """Root-to-leaf frame names ending at ``event``, in the perf report's call_stack_full
    format (``_<n>`` suffixes stripped)."""
    frames = []
    while event is not None:
        name = event.get("name", "Unknown")
        frames.append(re.sub(r"_\d+", "", name[:256]))
        event = tree.events_by_uid.get(event.get("parent"))
    return str(frames[::-1])


def classify_idle_intervals(
    tree, thresholds=None, gpu_events=None, include_call_stack=False
):
    """Classify every idle gap on the merged GPU timeline.

    Args:
        tree: TraceLens tree (``events``, ``events_by_uid``).
        thresholds: IdleThresholds; defaults if None.
        gpu_events: GPU events defining the timeline; computed with GPUEventAnalyser
            if None.
        include_call_stack: fill sync_entry_point and sync_call_stack_full for gaps
            with a sync call, and covering_call_stack_full for gaps with a covering op
            (same format as the perf report's call stack columns).

    Returns a list of dicts, one per gap in time order. Keys: start, end, duration,
    label_noise, case ('unqueued' | 'queued' | 'no_launch'), label, sub_label, detail,
    the parts in PARTS (summing to duration), drain_type and sync_* fields (including
    sync_gain_us), context (covering_op, top_op, top_runtime, launch_parent_op,
    launch_tid, launch_to_start_us, launch_lead_us, launch_call_x_ref, other_thread_*),
    bounding GPU events and their streams, and UIDs for walking back into the tree.
    """
    th = thresholds or IdleThresholds()
    if gpu_events is None:
        gpu_events = GPUEventAnalyser(tree.events).get_gpu_event_lists()[
            GPUEventAnalyser.all_gpu_key
        ]
    if not gpu_events:
        return []

    gaps = extract_idle_intervals(gpu_events)
    rt_index = OverlapIndex(build_sorted_cpu_events(tree))
    thread_index = build_thread_index(tree)
    timeline = _GpuTimeline(build_gpu_kernel_map(tree))

    # Pass 1: bounding events and launch for every macro gap.
    results, pending = [], []
    for gap_start, gap_end in gaps:
        rec = _new_record(gap_start, gap_end)
        results.append(rec)
        if rec["duration"] < th.noise_us:
            rec["label_noise"] = True
            continue
        i, preceding, following = timeline.bounding_events(gap_start, gap_end)
        launch = (
            find_launch_for_kernel(tree, following) if following is not None else None
        )
        _set_bounds(rec, preceding, following)
        _set_launch(rec, tree, launch)
        _find_sync_drain(rec, rt_index, gap_start, launch, timeline.by_end, i, th)
        pending.append((rec, launch))

    # Pass 2: parts and labels.
    limit = _launch_limit(pending, th)
    for rec, launch in pending:
        _measure_and_label(rec, launch, limit, th, thread_index, rt_index)
        if include_call_stack:
            _attach_call_stacks(rec, tree)
    return results


class _GpuTimeline:
    """GPU ops sorted by start and by end, for finding the ops bounding a gap."""

    def __init__(self, gpu_sorted):
        self.by_start = gpu_sorted
        self.starts = [e["ts"] for e in gpu_sorted]
        self.by_end = sorted(gpu_sorted, key=_t_end)
        self.ends = [_t_end(e) for e in self.by_end]

    def bounding_events(self, gap_start, gap_end):
        """(index of the preceding op in by_end or -1, preceding op, following op)."""
        i = bisect.bisect_right(self.ends, gap_start + 0.5) - 1
        preceding = None
        if i >= 0 and abs(self.ends[i] - gap_start) < 2.0:
            preceding = self.by_end[i]
        j = bisect.bisect_left(self.starts, gap_end - 0.001)
        following = None
        for k in range(max(0, j - 1), min(len(self.by_start), j + 3)):
            if abs(self.by_start[k]["ts"] - gap_end) < 1.0:
                following = self.by_start[k]
                break
        return (i if preceding is not None else -1), preceding, following


def _set_bounds(rec, preceding, following):
    if preceding is not None:
        rec["preceding_gpu_event"] = preceding.get("name")
        rec["preceding_stream"] = stream_of(preceding)
        rec["preceding_gpu_uid"] = preceding.get("UID")
    if following is not None:
        rec["following_gpu_event"] = following.get("name")
        rec["following_stream"] = stream_of(following)
        rec["following_gpu_uid"] = following.get("UID")
    if preceding is not None and following is not None:
        rec["stream_switch"] = rec["preceding_stream"] != rec["following_stream"]


def _set_launch(rec, tree, launch):
    if launch is None:
        return
    launch_end = _t_end(launch)
    rec["following_launch_name"] = launch.get("name")
    rec["following_launch_uid"] = launch.get("UID")
    rec["launch_parent_op"] = find_launch_parent_op(tree, launch)
    rec["launch_tid"] = launch.get("tid")
    rec["launch_to_start_us"] = rec["end"] - launch_end
    rec["launch_lead_us"] = rec["start"] - launch_end


def _launch_limit(pending, th):
    """Launch return -> GPU op start above which a gap counts as a GPU wait."""
    limit = th.launch_latency_us
    if th.launch_latency_rel is None:
        return limit
    during = [
        r["launch_to_start_us"]
        for r, launch in pending
        if launch is not None and r["start"] <= _t_end(launch) <= r["end"]
    ]
    if during:
        limit = max(limit, th.launch_latency_rel * statistics.median(during))
    return limit


def _attach_call_stacks(rec, tree):
    if rec["covering_op_uid"] is not None:
        rec["covering_call_stack_full"] = call_stack_full(
            tree, tree.events_by_uid.get(rec["covering_op_uid"])
        )
    if rec["sync_event_uid"] is not None:
        # Lazy import: the perf report module is heavy and imports this package.
        from TraceLens.Reporting.generate_perf_report_pytorch import (
            _find_entry_point,
        )

        stack = call_stack_full(tree, tree.events_by_uid.get(rec["sync_event_uid"]))
        rec["sync_call_stack_full"] = stack
        rec["sync_entry_point"] = _find_entry_point(stack, rec["sync_event_name"])[
            "entry_point"
        ]


def _measure_and_label(rec, launch, limit, th, thread_index, rt_index):
    gs, ge, d = rec["start"], rec["end"], rec["duration"]

    if launch is None:
        rec.update(case="no_launch", label="UNATTRIBUTED", unattributed_us=d)
        rec["detail"] = "no traced launch for the next GPU op"
        return

    thread = (launch.get("pid"), launch.get("tid"))
    _set_other_thread(rec, rt_index, thread)
    _, _, covering = exclusive_split(thread_index, thread, gs, ge, return_event=True)
    if covering is not None:
        rec["covering_op"] = strip_addresses(covering.get("name", "?"))
        rec["covering_op_uid"] = covering.get("UID")
    launch_name = launch.get("name", "")
    # Copies and memsets can block on the GPU, so only kernel and graph launches.
    if launch_name in LAUNCH_NAMES or "Launch" in launch_name:
        rec["launch_call_x_ref"] = launch.get("dur", 0.0) / th.launch_call_ref_us

    launch_end = _t_end(launch)
    if launch_end + limit <= gs:
        _label_queued(rec)
        return

    rec["case"] = "unqueued"
    late_end = min(max(launch_end, gs), ge)
    call_start = min(max(launch["ts"], gs), late_end)
    split, names, _ = exclusive_split(
        thread_index, thread, gs, call_start, launch.get("UID")
    )
    rec.update(split)
    if is_enqueue_call(launch_name):
        rec["launch_call_us"] = late_end - call_start
    else:
        # Blocking copies (hipMemcpyWithStream from .item()/.to()) wait on the stream.
        rec["cpu_runtime_us"] += late_end - call_start
        names[("cpu_runtime_us", launch_name)] += late_end - call_start
    rec["submit_latency_us"] = ge - late_end
    rec["top_op"] = strip_addresses(_top_name(names, "cpu_op_us"))
    rec["top_runtime"] = _top_name(names, "cpu_runtime_us")
    rec["label"], rec["sub_label"] = _unqueued_label(rec, th, limit)
    rec["detail"] = _fmt_parts(rec)


def _set_other_thread(rec, rt_index, thread):
    """The runtime call with the most time in the gap on a thread other than ``thread``."""
    gs, ge = rec["start"], rec["end"]
    other = defaultdict(float)
    for rt in rt_index.query(gs, ge):
        if (rt.get("pid"), rt.get("tid")) != thread:
            other[rt.get("name", "?")] += min(_t_end(rt), ge) - max(rt["ts"], gs)
    if other:
        top = max(other, key=other.get)
        rec["other_thread_top_runtime"] = top
        rec["other_thread_runtime_us"] = other[top]


def _label_queued(rec):
    rec.update(
        case="queued",
        label="UNEXPLAINED_GPU_WAIT",
        sub_label="queued",
        gpu_wait_us=rec["duration"],
    )
    rec["detail"] = (
        f"next op launched {rec['launch_lead_us']:.1f}us before the gap; CPU was not "
        "the bottleneck; cause is below the application (runtime, driver, firmware, "
        "hardware) or profiler overhead"
    )


def _unqueued_label(rec, th, limit):
    """(label, sub_label) of an unqueued gap: its largest part, with two overrides."""
    groups = {
        "CPU_OP_DOMINATED": rec["cpu_op_us"],
        "RUNTIME_DOMINATED": rec["cpu_runtime_us"],
        "UNTRACED_DOMINATED": rec["cpu_untraced_us"],
        "LAUNCH": rec["launch_call_us"] + rec["submit_latency_us"],
    }
    label = max(groups, key=groups.get)
    other_us = rec["other_thread_runtime_us"] or 0.0
    if (
        label != "RUNTIME_DOMINATED"
        and other_us > th.other_thread_frac * rec["duration"]
    ):
        return "RUNTIME_DOMINATED", "other_thread"
    if label != "LAUNCH":
        return label, None
    if rec["launch_to_start_us"] > limit:
        return "UNEXPLAINED_GPU_WAIT", "unqueued"
    return "LAUNCH_DOMINATED", None


# ---------------------------------------------------------------------------
# Augmented trace generation
# ---------------------------------------------------------------------------


def assign_idle_ids(classified):
    """Assign stable idle_id: macro intervals get 0-indexed IDs, noise gets negative."""
    macro_id = 0
    noise_id = -1
    for rec in classified:
        if rec["label_noise"]:
            rec["idle_id"] = noise_id
            noise_id -= 1
        else:
            rec["idle_id"] = macro_id
            macro_id += 1


def full_label(rec):
    return rec["label"] + (f"/{rec['sub_label']}" if rec.get("sub_label") else "")


ANNOTATION_TRACKS = (
    (9000, "Idle: Noise/Macro"),
    (9001, "Idle: Drain Type"),
    (9002, "Idle: Label"),
)


def make_annotation_events(classified, gpu_pid):
    """Chrome trace events for three annotation tracks on the GPU process."""
    events = []
    for tid, name in ANNOTATION_TRACKS:
        events.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": gpu_pid,
                "tid": tid,
                "args": {"name": name},
            }
        )
        events.append(
            {
                "ph": "M",
                "name": "thread_sort_index",
                "pid": gpu_pid,
                "tid": tid,
                "args": {"sort_index": tid},
            }
        )
    for rec in classified:
        events += _gap_slices(rec, gpu_pid)
    return events


def _slice(gpu_pid, tid, name, rec, args):
    return {
        "ph": "X",
        "cat": "idle_classification",
        "name": name,
        "pid": gpu_pid,
        "tid": tid,
        "ts": rec["start"],
        "dur": rec["duration"],
        "args": {"idle_id": rec["idle_id"], **args},
    }


def _gap_slices(rec, gpu_pid):
    """One slice per track for a macro gap; one on the first track for noise."""
    idle_id = rec["idle_id"]
    duration = {"duration_us": f"{rec['duration']:.2f}"}
    if rec["label_noise"]:
        return [_slice(gpu_pid, 9000, "noise", rec, duration)]
    if rec["drain_type"] == "sync_drain":
        drain = f"idle#{idle_id} sync_drain: {rec['sync_event_name']}"
    else:
        drain = f"idle#{idle_id} starved"
    drain_args = {
        "drain_type": rec["drain_type"],
        "sync_event_name": rec["sync_event_name"] or "none",
    }
    label_args = {
        "label": rec["label"],
        "sub_label": rec["sub_label"] or "",
        "case": rec["case"],
        "detail": rec["detail"] or "",
        **{p: round(rec[p], 2) for p in PARTS},
    }
    return [
        _slice(gpu_pid, 9000, f"idle#{idle_id}", rec, duration),
        _slice(gpu_pid, 9001, drain, rec, drain_args),
        _slice(gpu_pid, 9002, f"idle#{idle_id} {full_label(rec)}", rec, label_args),
    ]


def find_gpu_pid(events):
    """GPU process ID with the most kernel/memcpy/memset events."""
    from collections import Counter

    kernel_pids = Counter(
        e["pid"] for e in events if e.get("cat") in GPU_CATS and e.get("ph") == "X"
    )
    if kernel_pids:
        return kernel_pids.most_common(1)[0][0]
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_labels":
            if "GPU" in e.get("args", {}).get("labels", ""):
                return e["pid"]
    return 0
