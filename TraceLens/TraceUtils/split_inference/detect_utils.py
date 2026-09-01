###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared machinery for execution-root detection.

Four concerns that several detection steps each need, kept together so their
semantics are defined once:

- the result contract every step speaks (:class:`RootSet` and friends)
- containment queries over event spans (:class:`IntervalIndex`)
- attributing GPU kernels to annotations and measuring coverage
  (:class:`GpuAttribution`)
- enumerating alternative root candidates, and running escalation probes
"""

from bisect import bisect_left
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..annotation_utils import PROVENANCE_KEY, name_skeleton

# Projections *enclose* the kernels they describe, so summing GPU time over both
# double-counts. Kept apart here and recombined by consumers that want both.
GPU_KERNEL_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
PROJECTION_CATEGORY = "gpu_user_annotation"

# Coverage to accept roots outright, and the floor below which a trace is
# unsplittable rather than degraded.
COVERAGE_GATE = 0.95
COVERAGE_FLOOR = 0.75

# Least share of captured GPU time the annotation spans must explain themselves.
# Below this, coverage comes from window extension, so the roots are too sparse.
MIN_SPAN_SHARE = 0.5

# Fewer roots than this usually means only a warmup loop matched.
MIN_ROOTS = 8


# --- result contract --------------------------------------------------------
class DetectStatus(IntEnum):
    """Whether the trace can be split. Deliberately separate from phase trust."""

    SPLITTABLE = 0
    NOT_SPLITTABLE = 1
    DEGRADED = 2


class PhaseConfidence(str, Enum):
    """How much to trust the phase and batch-size labels on the roots."""

    HIGH = "high"  # parsed from a recognized annotation
    LOW = "low"  # inherited onto a synthetic root
    UNKNOWN = "unknown"  # derived from kernel or python-frame periodicity


@dataclass
class CoverageReport:
    """Result of a GPU-time coverage audit.

    ``covered_selected`` measures the roots' extraction windows, gaps included,
    since that is what the output contains; ``covered_spans`` measures the bare
    annotation spans. Judging on bare spans alone hunts for extra roots whenever
    an iteration's sampling step runs just outside its annotation.
    """

    strategy: str
    covered_any: float
    covered_selected: float
    covered_spans: float
    gpu_busy: float
    window: Optional[Tuple[float, float]] = None

    @property
    def span_share(self) -> float:
        """How much of the captured work the annotations themselves explain.

        Near 0 means a few annotations stretched over many iterations, which
        would pass a coverage check while bundling iterations into each slice.
        """
        if self.covered_selected <= 0:
            return 0.0
        return min(1.0, self.covered_spans / self.covered_selected)

    @property
    def passes(self) -> bool:
        """Whether the roots explain enough GPU work, without being stretched.

        Gating on the roots rather than on all annotations is what makes this a
        real check: a run whose annotations blanket the timeline while the roots
        cover fifteen of five hundred iterations scores near-perfectly on the
        permissive measure.
        """
        return self.covered_selected >= COVERAGE_GATE and (
            self.span_share >= MIN_SPAN_SHARE
        )

    @property
    def better_roots_exist(self) -> bool:
        """Annotations cover work the roots miss, so escalation should help."""
        return self.covered_any - self.covered_selected > 1 - COVERAGE_GATE


@dataclass
class RootSet:
    """Roots plus how they were found and how much we trust them."""

    roots: List[dict]
    method: str
    phase_confidence: PhaseConfidence = PhaseConfidence.UNKNOWN
    status: DetectStatus = DetectStatus.SPLITTABLE
    coverage: Optional[CoverageReport] = None
    diagnostics: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.roots)

    def to_manifest(self) -> dict:
        cov = self.coverage
        manifest = {
            "status": int(self.status),
            "method": self.method,
            "phase_confidence": self.phase_confidence.value,
            "n_roots": len(self.roots),
            "attribution_strategy": cov.strategy if cov else None,
            "coverage_any_annotation": round(cov.covered_any, 4) if cov else None,
            "coverage_selected_roots": round(cov.covered_selected, 4) if cov else None,
            "coverage_root_spans_only": round(cov.covered_spans, 4) if cov else None,
            "root_span_share": round(cov.span_share, 4) if cov else None,
        }
        manifest.update(self.diagnostics)
        return manifest


# --- containment queries ----------------------------------------------------
class IntervalIndex:
    """Event spans grouped by ``(pid, tid)`` and sorted by start timestamp.

    Grouping by thread keeps containment from crossing threads and inventing a
    false parent; sorting makes the query a bisect rather than a full scan.
    """

    def __init__(self, events: Iterable[dict]):
        buckets: Dict[Tuple, List[dict]] = {}
        for e in events:
            ts, dur = e.get("ts"), e.get("dur")
            if ts is None or dur is None:
                continue
            buckets.setdefault((e.get("pid"), e.get("tid")), []).append(e)
        self._threads: Dict[Tuple, Tuple[List[float], List[float], List[dict]]] = {}
        for key, evs in buckets.items():
            evs.sort(key=lambda x: x["ts"])
            starts = [x["ts"] for x in evs]
            ends = [x["ts"] + x["dur"] for x in evs]
            self._threads[key] = (starts, ends, evs)

    def contained_in(self, span: dict, exclude_self: bool = True) -> List[dict]:
        """Events lying entirely within ``span``, on ``span``'s own thread."""
        entry = self._threads.get((span.get("pid"), span.get("tid")))
        if entry is None:
            return []
        starts, ends, evs = entry
        start = span.get("ts", 0)
        end = start + span.get("dur", 0)
        out = []
        i = bisect_left(starts, start)
        while i < len(starts) and starts[i] < end:
            if ends[i] <= end and not (exclude_self and evs[i] is span):
                out.append(evs[i])
            i += 1
        return out


def gaps_between(roots: Sequence[dict]) -> List[dict]:
    """Idle spans between consecutive roots, per thread, as span dicts."""
    gaps = []
    for (pid, tid), group in group_by_thread(roots).items():
        for prev, nxt in zip(group, group[1:]):
            prev_end = prev.get("ts", 0) + prev.get("dur", 0)
            if nxt.get("ts", 0) > prev_end:
                gaps.append(
                    {
                        "ts": prev_end,
                        "dur": nxt["ts"] - prev_end,
                        "pid": pid,
                        "tid": tid,
                    }
                )
    return gaps


def build_root_tiles(roots: Sequence[dict]) -> Tuple[dict, int]:
    """Gap-free extraction windows, one per root, keyed by ``(pid, tid, ts)``.

    Each window reaches to the next root's start on the same thread, so work
    between two roots belongs to somebody; the last gets the median length
    rather than running to the end of the trace and swallowing teardown.

    Grouping by thread is required, not tidy: a global sort interleaves threads
    and a window spanning two of them describes nothing. Overlapping roots keep
    their own span and are counted instead.
    """
    tiles: dict = {}
    overlaps = 0
    for (pid, tid), group in group_by_thread(roots).items():
        spans = []
        for index, root in enumerate(group):
            start = root.get("ts", 0)
            own_end = start + root.get("dur", 0)
            end = own_end
            if index + 1 < len(group):
                following = group[index + 1].get("ts", 0)
                if following < own_end:
                    overlaps += 1
                else:
                    end = following
            spans.append((start, end))
        if len(spans) > 1:
            typical = median(end - start for start, end in spans[:-1])
            last_start, last_end = spans[-1]
            spans[-1] = (last_start, max(last_end, last_start + typical))
        for start, end in spans:
            tiles[(pid, tid, start)] = (start, end)
    return tiles, overlaps


def group_by_thread(events: Iterable[dict]) -> Dict[Tuple, List[dict]]:
    """Events grouped by ``(pid, tid)`` and sorted by timestamp within group."""
    groups: Dict[Tuple, List[dict]] = {}
    for e in events:
        groups.setdefault((e.get("pid"), e.get("tid")), []).append(e)
    for group in groups.values():
        group.sort(key=lambda x: x.get("ts", 0))
    return groups


class SpanSet:
    """A disjoint, sorted union of ``(start, end)`` time spans.

    Overlapping input is the norm -- annotations nest, projections repeat per
    stream -- so merging on construction is what makes membership a bisect.
    """

    def __init__(self, spans: Iterable[Tuple[float, float]] = ()):
        self.spans: List[Tuple[float, float]] = []
        for start, end in sorted(spans):
            if self.spans and start <= self.spans[-1][1]:
                last_start, last_end = self.spans[-1]
                self.spans[-1] = (last_start, max(last_end, end))
            else:
                self.spans.append((start, end))

    @classmethod
    def of_events(cls, events: Iterable[dict]) -> "SpanSet":
        return cls((e["ts"], e["ts"] + e["dur"]) for e in events)

    def __or__(self, other: "SpanSet") -> "SpanSet":
        return SpanSet(self.spans + other.spans)

    def __bool__(self) -> bool:
        return bool(self.spans)

    def covers(self, point: float) -> bool:
        i = bisect_left(self.spans, (point, float("inf")))
        if i > 0 and self.spans[i - 1][0] <= point <= self.spans[i - 1][1]:
            return True
        return i < len(self.spans) and self.spans[i][0] <= point <= self.spans[i][1]

    @property
    def bounds(self) -> Optional[Tuple[float, float]]:
        """Time frame enclosing every span, or ``None`` when there are none."""
        if not self.spans:
            return None
        return self.spans[0][0], max(end for _, end in self.spans)


# --- GPU attribution and coverage -------------------------------------------
class GpuAttribution:
    """Attributes GPU kernels to annotations and measures coverage.

    Two strategies, chosen per trace. ``projection`` (the kernel starts inside a
    ``gpu_user_annotation`` span) is preferred: cheaper, and immune to graph
    capture, where correlations cannot be walked at all. ``correlation`` (the
    launch traces back to a CPU op inside an annotation) is a mandatory
    fallback, since some traces have no projections -- notably any trace that is
    itself a previous split output.

    Attribution to *any* annotation counts, not just the selected root: the
    question is "are we missing whole regions of work", not fine accounting.
    """

    STRATEGY_PROJECTION = "projection"
    STRATEGY_CORRELATION = "correlation"

    def __init__(self, events: Iterable[dict]):
        self.kernels: List[dict] = []
        self.projections: List[dict] = []
        corr_cpu: List[dict] = []
        self._corr_kernels: Dict[int, List[dict]] = {}
        self._launch_by_corr: Dict[int, dict] = {}
        for e in events:
            ts, dur, cat = e.get("ts"), e.get("dur"), e.get("cat")
            if ts is None or dur is None:
                continue
            corr = (e.get("args") or {}).get("correlation")
            if cat == PROJECTION_CATEGORY:
                self.projections.append(e)
            elif cat in GPU_KERNEL_CATEGORIES:
                self.kernels.append(e)
                if corr is not None:
                    self._corr_kernels.setdefault(corr, []).append(e)
            elif corr is not None:
                corr_cpu.append(e)
                self._launch_by_corr.setdefault(corr, e)

        self.kernels.sort(key=lambda x: x["ts"])
        self._kernel_starts = [k["ts"] for k in self.kernels]
        self.strategy = (
            self.STRATEGY_PROJECTION if self.projections else self.STRATEGY_CORRELATION
        )
        # Built on demand: the projection path normally never needs it, but it
        # remains reachable as a cross-check when projections look untrustworthy.
        self._corr_cpu = corr_cpu
        self._cpu_index_cache: Optional[IntervalIndex] = None

    @property
    def _cpu_index(self) -> IntervalIndex:
        if self._cpu_index_cache is None:
            self._cpu_index_cache = IntervalIndex(self._corr_cpu)
        return self._cpu_index_cache

    def _kernels_in(self, window: Optional[Tuple[float, float]]) -> List[dict]:
        """Kernels whose *start* lies in ``window``.

        Selecting on start rather than clipping durations keeps coverage from
        exceeding 1 through partially-overlapping kernels.
        """
        if window is None:
            return self.kernels
        lo, hi = window
        i = bisect_left(self._kernel_starts, lo)
        out = []
        while i < len(self.kernels) and self._kernel_starts[i] <= hi:
            out.append(self.kernels[i])
            i += 1
        return out

    def kernels_for(self, spans: Sequence[dict]) -> List[dict]:
        """Kernels launched from CPU ops inside ``spans`` (correlation path)."""
        seen, out = set(), []
        for span in spans:
            for cpu in self._cpu_index.contained_in(span, exclude_self=False):
                corr = (cpu.get("args") or {}).get("correlation")
                for k in self._corr_kernels.get(corr, ()):
                    if id(k) not in seen:
                        seen.add(id(k))
                        out.append(k)
        return out

    def cpu_launches_for(self, kernels: Sequence[dict]) -> List[dict]:
        """CPU launch sites of ``kernels``, empty under graph capture."""
        seen, out = set(), []
        for kernel in kernels:
            corr = (kernel.get("args") or {}).get("correlation")
            launch = self._launch_by_corr.get(corr)
            if launch is not None and id(launch) not in seen:
                seen.add(id(launch))
                out.append(launch)
        return out

    def gpu_time_by_correlation(self, spans: Sequence[dict]) -> float:
        """GPU time launched from CPU ops inside ``spans``."""
        return sum(k["dur"] for k in self.kernels_for(spans))

    def gpu_time_for_family(self, skeleton: str, instances: Sequence[dict]) -> float:
        """GPU time attributable to one annotation family."""
        if self.strategy != self.STRATEGY_PROJECTION:
            return self.gpu_time_by_correlation(instances)
        spans = self._projection_union({skeleton})
        if not spans:
            return 0.0
        return sum(
            k["dur"] for k in self._kernels_in(spans.bounds) if spans.covers(k["ts"])
        )

    # -- coverage ------------------------------------------------------------
    def audit(
        self, annotations: Sequence[dict], roots: Sequence[dict]
    ) -> CoverageReport:
        """Measure GPU coverage by all annotations, and by the roots alone.

        High ``covered_any`` next to low ``covered_selected`` means the roots
        sit at the wrong nesting level, and widening should fix it.
        """
        # Credit the roots two ways: projections survive graph capture, and
        # launch correlations work for roots no annotation produced. Matching by
        # name alone scores every synthetic root zero and stalls escalation.
        if self.strategy == self.STRATEGY_PROJECTION:
            by_name = self._projection_union(_window_names(roots))
            any_spans = self._projection_union(None)
            window = SpanSet.of_events(self.projections).bounds
        else:
            by_name = SpanSet()
            any_spans = SpanSet.of_events(self.kernels_for(annotations))
            window = any_spans.bounds

        in_window = self._kernels_in(window)
        busy = sum(k["dur"] for k in in_window)
        if busy <= 0:
            return CoverageReport(self.strategy, 0.0, 0.0, 0.0, 0.0, window)

        tiles, _ = build_root_tiles(roots)
        tile_spans = [
            {"pid": pid, "tid": tid, "ts": start, "dur": end - start}
            for (pid, tid, _), (start, end) in tiles.items()
        ]
        covered = {}
        for label, spans in (
            ("any", any_spans),
            ("spans", by_name | SpanSet.of_events(self.kernels_for(roots))),
            ("tiles", by_name | SpanSet.of_events(self.kernels_for(tile_spans))),
        ):
            covered[label] = sum(k["dur"] for k in in_window if spans.covers(k["ts"]))
        return CoverageReport(
            self.strategy,
            covered["any"] / busy,
            covered["tiles"] / busy,
            covered["spans"] / busy,
            busy,
            window,
        )

    def uncovered_kernels(self, annotations: Sequence[dict]) -> List[dict]:
        """Kernels in the audit window that no annotation accounts for."""
        if self.strategy == self.STRATEGY_PROJECTION:
            spans = self._projection_union(None)
            window = SpanSet.of_events(self.projections).bounds
        else:
            spans = SpanSet.of_events(self.kernels_for(annotations))
            window = spans.bounds
        return [k for k in self._kernels_in(window) if not spans.covers(k["ts"])]

    def _projection_union(self, names: Optional[set]) -> SpanSet:
        """Union of projection spans, optionally restricted by annotation name.

        Taken across every GPU thread rather than per stream: a kernel on a side
        stream inside an annotated region is genuinely covered, and per-stream
        matching would call it uncovered and escalate for nothing.
        """
        return SpanSet.of_events(
            p
            for p in self.projections
            if names is None or name_skeleton(p.get("name", "")) in names
        )


def _window_names(roots: Sequence[dict]) -> set:
    """Skeletons to match projections against for the selected roots.

    A root enriched in step 1.5 carries the inner annotation's name, so its
    projections are recorded under the outer span it actually came from.
    """
    names = set()
    for r in roots:
        prov = r.get(PROVENANCE_KEY) or {}
        names.add(name_skeleton(prov.get("window_from") or r.get("name", "")))
    return names


# Escalation probes were removed: on the fallback corpus they never once improved
# a split, and the coverage gate now grades roots directly (splittable / degraded
# / not-splittable) with no intermediate probe ladder.
