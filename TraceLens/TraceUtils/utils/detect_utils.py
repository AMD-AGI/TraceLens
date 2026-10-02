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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ...util import GPU_KERNEL_CATEGORIES, GPU_USER_ANNOTATION

# Coverage to accept roots outright, and the floor below which a trace is
# unsplittable rather than degraded.
COVERAGE_GATE = 0.95
COVERAGE_FLOOR = 0.75

# Least share of captured GPU time the annotation spans must explain themselves.
# Below this, coverage comes from window extension, so the roots are too sparse.
MIN_SPAN_SHARE = 0.5

# Fewer roots than this usually means only a warmup loop matched.
MIN_ROOTS = 4


# --- result contract --------------------------------------------------------
class DetectStatus(IntEnum):
    """Whether the trace can be split. Deliberately separate from phase trust."""

    SPLITTABLE = 0
    NOT_SPLITTABLE = 2
    DEGRADED = 1


class PhaseConfidence(str, Enum):
    """How much to trust the phase and batch-size labels on the roots."""

    HIGH = "high"  # parsed from a recognized annotation
    LOW = "low"  # inherited onto a synthetic root
    UNKNOWN = "unknown"  # derived from kernel or python-frame periodicity


@dataclass
class CoverageReport:
    """Result of a GPU-time coverage audit.

    ``covered_selected`` measures the roots' extraction windows, gaps included; ``covered_spans`` measures the bare annotation spans.
    """

    strategy: str
    covered_selected: float
    covered_spans: float
    gpu_busy: float

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
        """Whether the roots explain enough GPU work, without being stretched."""
        return self.covered_selected >= COVERAGE_GATE and (
            self.span_share >= MIN_SPAN_SHARE
        )


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
            "coverage_selected_roots": round(cov.covered_selected, 4) if cov else None,
            "coverage_root_spans_only": round(cov.covered_spans, 4) if cov else None,
            "root_span_share": round(cov.span_share, 4) if cov else None,
        }
        # Underscore keys are objects passed between stages -- the event map, for
        # one -- not findings. Serializing them would swamp the manifest.
        manifest.update(
            {k: v for k, v in self.diagnostics.items() if not k.startswith("_")}
        )
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
            if e.get("dur") is None:
                continue  # flow markers etc. have no span; skip rather than crash
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

    Overlapping input is the norm -- annotations nest, GPU annotation spans
    repeat per stream -- so merging on construction makes membership a bisect.
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


# --- Single-pass event index ------------------------------------------------

ANNOTATION_CAT = "user_annotation"


@dataclass
class TraceData:
    """The parsed trace and its correlation maps, threaded through extraction.

    The immutable inputs every extraction call needs: the raw event list, the
    trace JSON to clone into each output, and the correlation maps that link CPU
    launches to GPU work. Output location and gap-fill tiles are *not* here --
    those are per-call parameters, not trace data.
    """

    events: List[dict]
    trace_json: dict
    gpu_corr_map: dict
    flow_corr_map: dict
    meta_events: List[dict]


@dataclass
class ExtractContext:
    """A write job shared by every ``extract_and_save_*`` call in a run.

    Bundles the trace with its output target so the writers take one context
    instead of the same four arguments each. ``output_dir`` is the base directory
    (``divide_phases_and_save`` writes into per-phase subfolders of it), and
    ``root_tiles`` are the gap-free extraction windows over the whole root list.
    """

    trace: TraceData
    output_dir: str
    base_name: str
    root_tiles: Optional[dict] = None


class EventIndex:
    """All event categories extracted in a single pass over the trace."""

    def __init__(self, events: Iterable[dict]):
        self.kernels: List[dict] = []
        self.annotations: List[dict] = []
        self.gpu_annotation_spans: List[dict] = []
        self.meta_events: List[dict] = []
        self.gpu_corr_map: Dict = {}
        self.flow_corr_map: Dict = {}
        self.corr_cpu: List[dict] = []

        for e in events:
            ts = e.get("ts")
            cat = e.get("cat")

            if ts is None:
                self.meta_events.append(e)
                continue

            ph = e.get("ph")
            if ph in ("s", "f"):
                corr = e.get("id")
                if corr is not None:
                    self.flow_corr_map.setdefault(corr, []).append(e)
                continue

            dur = e.get("dur")
            if dur is None:
                continue

            if cat == ANNOTATION_CAT:
                if "Input Dims" not in (e.get("args") or {}):
                    self.annotations.append(e)
                continue

            corr = (e.get("args") or {}).get("correlation")

            if cat == GPU_USER_ANNOTATION:
                self.gpu_annotation_spans.append(e)
                if corr is not None:
                    self.gpu_corr_map.setdefault(corr, []).append(e)
            elif cat in GPU_KERNEL_CATEGORIES:
                self.kernels.append(e)
                if corr is not None:
                    self.gpu_corr_map.setdefault(corr, []).append(e)
            elif corr is not None:
                self.corr_cpu.append(e)

        self.kernels.sort(key=lambda x: x["ts"])
        self.annotations.sort(key=lambda e: e["ts"])


# --- GPU attribution and coverage -------------------------------------------
class GpuAttribution:
    """Attributes GPU kernels to annotations and measures coverage.

    Two strategies, chosen per set of instances
    ``gpu_span`` (the kernel starts inside a ``gpu_user_annotation`` span) is
    preferred because it needs no launch link, and applies when *every* instance
    has a GPU counterpart: one instance missing its span silently undercounts the
    whole set, so a single miss sends the set to ``correlation`` (the launch
    traces back to a CPU op inside the instance).
    """

    STRATEGY_GPU_SPAN = "gpu_span"
    STRATEGY_CORRELATION = "correlation"

    def __init__(self, source: EventIndex):
        self.annotations = source.annotations
        self.kernels = source.kernels
        self.gpu_annotation_spans = source.gpu_annotation_spans

        self._kernel_starts = [k["ts"] for k in self.kernels]
        self.gpu_busy = sum(k["dur"] for k in self.kernels)

        self._corr_kernels: Dict[int, List[dict]] = {}
        for k in self.kernels:
            corr = (k.get("args") or {}).get("correlation")
            if corr is not None:
                self._corr_kernels.setdefault(corr, []).append(k)

        self._corr_cpu = source.corr_cpu
        self._cpu_index_cache: Optional[IntervalIndex] = None
        self._spans_by_external_id: Dict[object, List[dict]] = {}
        for span in self.gpu_annotation_spans:
            ext = (span.get("args") or {}).get("External id")
            if ext is not None:
                self._spans_by_external_id.setdefault(ext, []).append(span)

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

    def attributed_kernels(self, instances: Sequence[dict]) -> Tuple[List[dict], str]:
        """Kernels belonging to ``instances``, and which strategy found them."""
        matched = [
            self._spans_by_external_id.get((e.get("args") or {}).get("External id"))
            for e in instances
        ]
        if instances and all(matched):
            spans = SpanSet.of_events([s for group in matched for s in group])
            return [
                k for k in self._kernels_in(spans.bounds) if spans.covers(k["ts"])
            ], self.STRATEGY_GPU_SPAN
        return self.kernels_for(instances), self.STRATEGY_CORRELATION

    def gpu_time_for_family(self, instances: Sequence[dict]) -> float:
        """GPU time attributable to one annotation family."""
        return sum(k["dur"] for k in self.attributed_kernels(instances)[0])

    def audit(self, roots: Sequence[dict]) -> CoverageReport:
        """Measure what share of the trace's GPU time the roots account for."""
        root_kernels, strategy = self.attributed_kernels(roots)
        if self.gpu_busy <= 0:
            return CoverageReport(strategy, 0.0, 0.0, self.gpu_busy)

        # Extraction hands out whole windows, so judge the window: first root's
        # start to the last one's end, per thread
        windows = [
            {
                "pid": pid,
                "tid": tid,
                "ts": group[0].get("ts", 0),
                "dur": group[-1].get("ts", 0)
                + group[-1].get("dur", 0)
                - group[0].get("ts", 0),
            }
            for (pid, tid), group in group_by_thread(roots).items()
        ]
        selected = {id(k): k for k in root_kernels}
        selected.update({id(k): k for k in self.kernels_for(windows)})
        return CoverageReport(
            strategy,
            sum(k["dur"] for k in selected.values()) / self.gpu_busy,
            sum(k["dur"] for k in root_kernels) / self.gpu_busy,
            self.gpu_busy,
        )
