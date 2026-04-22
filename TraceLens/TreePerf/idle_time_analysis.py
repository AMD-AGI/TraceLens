###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import bisect
from collections import defaultdict

import pandas as pd

from ..util import TraceEventUtils

_SYNC_PATTERNS = frozenset(
    {
        "cudadevicesynchronize",
        "cudastreamsynchronize",
        "cudaeventsynchronize",
        "hipdevicesynchronize",
        "hipstreamsynchronize",
        "hipeventsynchronize",
    }
)

_STREAM_WAIT_PATTERNS = frozenset(
    {
        "cudastreamwaitevent",
        "hipstreamwaitevent",
    }
)

_MEMORY_PATTERNS = frozenset(
    {
        "cudamemcpy",
        "cudamemcpyasync",
        "cudamemset",
        "cudamemsetasync",
        "cudamalloc",
        "cudamallocasync",
        "cudafree",
        "cudafreeasync",
        "hipmemcpy",
        "hipmemcpyasync",
        "hipmemset",
        "hipmemsetasync",
        "hipmalloc",
        "hipmallocasync",
        "hipfree",
        "hipfreeasync",
        "cudamallocmanaged",
        "cudamemcpy2d",
        "cudamemcpy2dasync",
        "cudamemcpy3d",
        "cudamemcpy3dasync",
    }
)

REASON_EXPLICIT_SYNC = "explicit_sync"
REASON_CROSS_STREAM_DEP = "cross_stream_dep"
REASON_CPU_BOTTLENECK = "cpu_bottleneck"
REASON_LAUNCH_OVERHEAD = "launch_overhead"
REASON_MEMORY_OP_STALL = "memory_op_stall"
REASON_SCHEDULER_SATURATION = "scheduler_saturation"
REASON_UNKNOWN = "unknown"


def _name_lower(event):
    return event.get("name", "").lower()


def _matches_any(name_lower, patterns):
    return any(name_lower.startswith(p) for p in patterns)


class IdleTimeAnalyser:
    """Classifies every idle gap on GPU stream(s) by root cause.

    Requires a *built* TraceToTree instance (build_tree() already called)
    so that parent/children links, per-stream ordering, and GPU-to-runtime
    correlation are all in place.
    """

    def __init__(
        self,
        tree,
        event_to_category=None,
        launch_overhead_thresh_us=10.0,
    ):
        self.tree = tree
        self.event_to_category = (
            event_to_category or TraceEventUtils.default_categorizer
        )
        self.launch_overhead_thresh_us = launch_overhead_thresh_us
        self._runtime_index_built = False
        self._all_runtime_sorted = []
        self._runtime_ts_keys = []

    def _build_runtime_event_index(self):
        """Pre-sort all cuda_runtime/cuda_driver events by ts for fast interval queries."""
        if self._runtime_index_built:
            return

        runtime_cats = {"cuda_runtime", "cuda_driver"}
        runtime_events = []
        for event in self.tree.events:
            cat = self.event_to_category(event)
            if cat in runtime_cats:
                if "t_end" not in event:
                    ts = event.get("ts")
                    dur = event.get("dur", 0)
                    if ts is not None:
                        event["t_end"] = ts + dur
                if event.get("ts") is not None:
                    runtime_events.append(event)

        runtime_events.sort(key=lambda e: e["ts"])
        self._all_runtime_sorted = runtime_events
        self._runtime_ts_keys = [e["ts"] for e in runtime_events]
        self._runtime_index_built = True

    def _find_runtime_events_in_interval(self, t_start, t_end):
        """Find runtime events whose [ts, t_end] overlaps [t_start, t_end]."""
        self._build_runtime_event_index()
        if not self._all_runtime_sorted:
            return []

        # Events that could overlap must have ts < t_end.
        # Start scanning from events whose ts could overlap: ts < t_end.
        right = bisect.bisect_left(self._runtime_ts_keys, t_end)
        results = []
        for i in range(right - 1, -1, -1):
            evt = self._all_runtime_sorted[i]
            evt_end = evt.get("t_end", evt["ts"])
            if evt_end <= t_start:
                # Events are sorted by ts; once t_end <= t_start we can
                # keep scanning because earlier events might still overlap
                # if they have long durations. But in practice runtime
                # events are short, so use a generous cutoff.
                if evt["ts"] < t_start - 1_000_000:
                    break
                continue
            if evt["ts"] < t_end and evt_end > t_start:
                results.append(evt)
        return results

    def _get_launch_parent(self, gpu_event):
        """Return the cuda_runtime launch event that spawned this GPU kernel."""
        parent_uid = gpu_event.get("parent")
        if parent_uid is None:
            return None
        parent = self.tree.events_by_uid.get(parent_uid)
        if parent is None:
            return None
        cat = self.event_to_category(parent)
        if cat in {"cuda_runtime", "cuda_driver"}:
            return parent
        return None

    def _get_cpu_op_ancestor(self, event):
        """Walk up the parent chain to find the nearest cpu_op ancestor."""
        current_uid = event.get("parent")
        while current_uid is not None:
            parent = self.tree.events_by_uid.get(current_uid)
            if parent is None:
                return None
            if self.event_to_category(parent) == "cpu_op":
                return parent
            current_uid = parent.get("parent")
        return None

    def _get_per_stream_gaps(self, stream_id=None):
        """Enumerate gaps between consecutive GPU events on each stream."""
        if not hasattr(self.tree, "dict_stream_index2event"):
            return []

        stream_index_map = self.tree.dict_stream_index2event
        streams = defaultdict(list)
        for (stream, idx), event in stream_index_map.items():
            streams[stream].append((idx, event))

        gaps = []
        for stream, idx_events in streams.items():
            if stream_id is not None and stream != stream_id:
                continue
            idx_events.sort(key=lambda x: x[0])
            for i in range(len(idx_events) - 1):
                _, prev_event = idx_events[i]
                _, next_event = idx_events[i + 1]
                prev_end = prev_event.get("t_end", prev_event.get("ts", 0))
                next_start = next_event.get("ts", 0)
                duration = next_start - prev_end
                if duration <= 0:
                    continue
                gaps.append(
                    {
                        "stream": stream,
                        "gap_start": prev_end,
                        "gap_end": next_start,
                        "duration_us": duration,
                        "prev_event": prev_event,
                        "next_event": next_event,
                    }
                )
        gaps.sort(key=lambda g: g["gap_start"])
        return gaps

    @staticmethod
    def _same_thread(event_a, event_b):
        """True if two events share the same (pid, tid)."""
        if event_a is None or event_b is None:
            return False
        return (
            event_a.get("pid") == event_b.get("pid")
            and event_a.get("tid") == event_b.get("tid")
        )

    def _classify_gap(self, gap_info):
        """Classify a single gap by examining CPU-side context.

        The key insight: first determine whether the CPU was "late" issuing
        the next launch.  If the launch was already queued before the gap
        started, the gap is GPU-side (pipeline latency or scheduler
        saturation) and we should NOT blame CPU-side events that merely
        happen to overlap temporally (e.g. a sync on a different logical
        path).  CPU-side investigation only matters when the launch was
        demonstrably delayed.
        """
        gap_start = gap_info["gap_start"]
        gap_end = gap_info["gap_end"]
        duration = gap_info["duration_us"]
        prev_event = gap_info["prev_event"]
        next_event = gap_info["next_event"]

        launch_event = self._get_launch_parent(next_event)
        cpu_op_ancestor = None
        if launch_event is not None:
            cpu_op_ancestor = self._get_cpu_op_ancestor(launch_event)

        # --- No launch event at all: the next GPU event (often a memset
        #     or memcpy) isn't linked to a CPU-side launch via ac2g
        #     correlation.  Check for GPU-side stream waits, then fall back.
        if launch_event is None:
            runtime_during_gap = self._find_runtime_events_in_interval(
                gap_start, gap_end
            )
            for rt_evt in runtime_during_gap:
                nl = _name_lower(rt_evt)
                if _matches_any(nl, _STREAM_WAIT_PATTERNS):
                    return self._build_result(
                        gap_info,
                        REASON_CROSS_STREAM_DEP,
                        f"Stream wait '{rt_evt.get('name')}' active during gap",
                        launch_event,
                        cpu_op_ancestor,
                    )
            next_cat = next_event.get("cat", "")
            next_name = next_event.get("name", "")
            return self._build_result(
                gap_info,
                REASON_UNKNOWN,
                (
                    f"No launch event linked for next GPU op "
                    f"(cat='{next_cat}', name='{next_name[:60]}')"
                ),
                launch_event,
                cpu_op_ancestor,
            )

        launch_ts = launch_event.get("ts", 0)
        launch_end = launch_event.get(
            "t_end", launch_ts + launch_event.get("dur", 0)
        )
        prev_event_end = prev_event.get("t_end", prev_event.get("ts", 0))

        # Was the launch already issued before this gap started?
        launch_was_pipelined = launch_ts <= gap_start

        # ------------------------------------------------------------------
        # PATH A: Launch was pipelined (CPU issued it before gap started).
        # The gap is GPU-side: either normal pipeline latency, a GPU-level
        # cross-stream wait, or scheduler saturation.
        # ------------------------------------------------------------------
        if launch_was_pipelined:
            # A1. Check for cross-stream dependency (GPU-side wait).
            #     hipStreamWaitEvent affects the GPU stream regardless of
            #     which CPU thread issued it.
            runtime_during_gap = self._find_runtime_events_in_interval(
                gap_start, gap_end
            )
            for rt_evt in runtime_during_gap:
                nl = _name_lower(rt_evt)
                if _matches_any(nl, _STREAM_WAIT_PATTERNS):
                    return self._build_result(
                        gap_info,
                        REASON_CROSS_STREAM_DEP,
                        f"Stream wait '{rt_evt.get('name')}' active during gap",
                        launch_event,
                        cpu_op_ancestor,
                    )

            # A2. Small gap → normal kernel launch overhead.
            if duration <= self.launch_overhead_thresh_us:
                return self._build_result(
                    gap_info,
                    REASON_LAUNCH_OVERHEAD,
                    (
                        f"Gap {duration:.1f}us <= threshold "
                        f"{self.launch_overhead_thresh_us:.1f}us; "
                        f"launch was pipelined"
                    ),
                    launch_event,
                    cpu_op_ancestor,
                )

            # A3. Larger gap with pipelined launch → scheduler saturation.
            next_kernel_start = next_event.get("ts", 0)
            queuing_delay = next_kernel_start - launch_end
            return self._build_result(
                gap_info,
                REASON_SCHEDULER_SATURATION,
                (
                    f"Launch completed {queuing_delay:.1f}us before kernel started; "
                    f"GPU scheduler likely saturated"
                ),
                launch_event,
                cpu_op_ancestor,
            )

        # ------------------------------------------------------------------
        # PATH B: Launch was NOT pipelined (CPU issued it after the gap
        # started → launch_ts > gap_start).  The CPU was late.
        # Investigate WHY: sync, memory op, or general CPU work.
        # ------------------------------------------------------------------
        runtime_during_gap = self._find_runtime_events_in_interval(
            gap_start, gap_end
        )

        # B1. Explicit synchronization on the launch thread.
        for rt_evt in runtime_during_gap:
            nl = _name_lower(rt_evt)
            if _matches_any(nl, _SYNC_PATTERNS):
                if self._same_thread(rt_evt, launch_event):
                    return self._build_result(
                        gap_info,
                        REASON_EXPLICIT_SYNC,
                        f"Sync API '{rt_evt.get('name')}' blocked launch thread during gap",
                        launch_event,
                        cpu_op_ancestor,
                    )

        # B2. Cross-stream dependency (GPU-side wait).
        for rt_evt in runtime_during_gap:
            nl = _name_lower(rt_evt)
            if _matches_any(nl, _STREAM_WAIT_PATTERNS):
                return self._build_result(
                    gap_info,
                    REASON_CROSS_STREAM_DEP,
                    f"Stream wait '{rt_evt.get('name')}' active during gap",
                    launch_event,
                    cpu_op_ancestor,
                )

        # B3. Memory operation stall on the launch thread.
        for rt_evt in runtime_during_gap:
            nl = _name_lower(rt_evt)
            if _matches_any(nl, _MEMORY_PATTERNS):
                if self._same_thread(rt_evt, launch_event):
                    return self._build_result(
                        gap_info,
                        REASON_MEMORY_OP_STALL,
                        (
                            f"Memory API '{rt_evt.get('name')}' "
                            f"blocked launch thread during gap"
                        ),
                        launch_event,
                        cpu_op_ancestor,
                    )

        # B4. General CPU bottleneck: the launch was late but no specific
        #     blocking API was found — CPU was busy with other work.
        cpu_delay = launch_ts - prev_event_end
        if cpu_op_ancestor:
            detail_msg = (
                f"CPU launch '{launch_event.get('name')}' started "
                f"{cpu_delay:.1f}us after prev kernel ended; "
                f"CPU op ancestor: '{cpu_op_ancestor.get('name')}'"
            )
        else:
            detail_msg = (
                f"CPU launch '{launch_event.get('name')}' started "
                f"{cpu_delay:.1f}us after prev kernel ended"
            )
        return self._build_result(
            gap_info,
            REASON_CPU_BOTTLENECK,
            detail_msg,
            launch_event,
            cpu_op_ancestor,
        )

    @staticmethod
    def _build_result(gap_info, reason, details, launch_event, cpu_op_ancestor):
        return {
            "stream": gap_info["stream"],
            "gap_start": gap_info["gap_start"],
            "gap_end": gap_info["gap_end"],
            "duration_us": gap_info["duration_us"],
            "reason": reason,
            "prev_gpu_event_uid": gap_info["prev_event"].get("UID"),
            "prev_gpu_event_name": gap_info["prev_event"].get("name"),
            "next_gpu_event_uid": gap_info["next_event"].get("UID"),
            "next_gpu_event_name": gap_info["next_event"].get("name"),
            "launch_event_uid": launch_event.get("UID") if launch_event else None,
            "launch_event_name": launch_event.get("name") if launch_event else None,
            "cpu_op_ancestor_uid": (
                cpu_op_ancestor.get("UID") if cpu_op_ancestor else None
            ),
            "cpu_op_ancestor_name": (
                cpu_op_ancestor.get("name") if cpu_op_ancestor else None
            ),
            "details": details,
        }

    def get_gaps_df(self, stream_id=None):
        """One row per idle gap with classification.

        Returns a DataFrame with columns: stream, gap_start, gap_end,
        duration_us, reason, prev/next GPU event info, launch event info,
        cpu_op ancestor info, and a human-readable details string.
        """
        self._build_runtime_event_index()
        raw_gaps = self._get_per_stream_gaps(stream_id=stream_id)
        classified = [self._classify_gap(g) for g in raw_gaps]
        if not classified:
            return pd.DataFrame(
                columns=[
                    "stream",
                    "gap_start",
                    "gap_end",
                    "duration_us",
                    "reason",
                    "prev_gpu_event_uid",
                    "prev_gpu_event_name",
                    "next_gpu_event_uid",
                    "next_gpu_event_name",
                    "launch_event_uid",
                    "launch_event_name",
                    "cpu_op_ancestor_uid",
                    "cpu_op_ancestor_name",
                    "details",
                ]
            )
        return pd.DataFrame(classified)

    def get_summary_df(self, stream_id=None):
        """Aggregated idle time by reason category.

        Returns a DataFrame with columns: reason, count, total_time_us,
        mean_duration_us, max_duration_us, pct_of_total_idle.
        """
        df = self.get_gaps_df(stream_id=stream_id)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "reason",
                    "count",
                    "total_time_us",
                    "mean_duration_us",
                    "max_duration_us",
                    "pct_of_total_idle",
                ]
            )

        total_idle = df["duration_us"].sum()
        summary = (
            df.groupby("reason")["duration_us"]
            .agg(["count", "sum", "mean", "max"])
            .reset_index()
        )
        summary.columns = [
            "reason",
            "count",
            "total_time_us",
            "mean_duration_us",
            "max_duration_us",
        ]
        summary["pct_of_total_idle"] = summary["total_time_us"] / total_idle * 100
        summary = summary.sort_values("total_time_us", ascending=False).reset_index(
            drop=True
        )
        return summary

    def get_top_gaps(self, n=10, stream_id=None):
        """The N largest idle gaps with their classifications."""
        df = self.get_gaps_df(stream_id=stream_id)
        if df.empty:
            return df
        return df.nlargest(n, "duration_us").reset_index(drop=True)
