###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the coverage-gated splitter: components and the flow built on them."""

import pytest

from TraceLens.TraceUtils.utils.annotation_utils import (
    cluster_by_skeleton,
    dominant_cluster,
    is_parseable,
    name_skeleton,
    parse_annotation,
)
from TraceLens.TraceUtils.trace_split import (
    DetectStatus,
    PhaseConfidence,
    TraceData,
    TraceIndex,
    build_root_tiles,
    extract_iteration,
    find_iteration_roots,
)
from TraceLens.TraceUtils.utils.detect_utils import (
    COVERAGE_GATE,
    GpuAttribution,
    IntervalIndex,
    group_by_thread,
)
from TraceLens.TraceUtils.trace_split.period_detection import (
    _find_repeating_period,
)


def collect_annotations(events):
    """The annotation events of a raw trace (was root_detection.collect_annotations)."""
    return TraceIndex(events).annotations


def _trace_data(events):
    """A ``TraceData`` for extraction, built from a raw event list."""
    ti = TraceIndex(events)
    return TraceData(
        events,
        {"traceEvents": events},
        ti.gpu_corr_map,
        ti.flow_corr_map,
        ti.meta_events,
    )


def _detect(events):
    """find_iteration_roots with the TraceIndex it now requires."""
    return find_iteration_roots(events, trace_index=TraceIndex(events))

VLLM = "execute_{i}_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"


# --------------------------------------------------------------------------- #
# Event builders
# --------------------------------------------------------------------------- #
def annotation(name, ts, dur, pid=1, tid=10, ext=None):
    return {
        "name": name,
        "cat": "user_annotation",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {} if ext is None else {"External id": ext},
    }


def launch(ts, corr, pid=1, tid=10, dur=2):
    return {
        "name": "hipLaunchKernel",
        "cat": "cuda_runtime",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {"correlation": corr},
    }


def kernel(ts, dur, corr, name="gemm", pid=1, tid=99):
    return {
        "name": name,
        "cat": "kernel",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {"correlation": corr},
    }


def gpu_annotation_span(name, ts, dur, pid=1, tid=99, ext=None):
    # Carries the External id its CPU annotation carries: that shared id is how
    # a GPU span is known to describe one specific instance.
    return {
        "name": name,
        "cat": "gpu_user_annotation",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {} if ext is None else {"External id": ext},
    }


def serving_trace(count=16, name_template=VLLM, period=1000, with_gpu_annotation=False):
    """One annotation per iteration, each launching one kernel."""
    events, corr = [], 500
    for i in range(count):
        base = 1000 + i * period
        name = name_template.format(i=i)
        events.append(annotation(name, base, 100, ext=i))
        events.append(launch(base + 10, corr))
        events.append(kernel(base + 200, 40, corr))
        if with_gpu_annotation:
            events.append(gpu_annotation_span(name, base + 200, 40, ext=i))
        corr += 1
    return events


# --------------------------------------------------------------------------- #
# C1: family keys
# --------------------------------------------------------------------------- #
class TestNameSkeleton:
    def test_collapses_digit_runs(self):
        assert (
            name_skeleton("execute_1_context_3(sq8sk8)")
            == "execute_#_context_#(sq#sk#)"
        )

    def test_instances_of_one_operation_share_a_key(self):
        a = name_skeleton("execute_1_context_3(sq128sk256)")
        b = name_skeleton("execute_77_context_9(sq4sk4)")
        assert a == b

    def test_distinguishes_genuinely_different_operations(self):
        assert name_skeleton("step[DECODE bs=4]") != name_skeleton("step[EXTEND bs=4]")

    def test_separates_families_differing_only_after_a_prefix(self):
        """A fixed-length prefix key would merge these; the skeleton must not."""
        long_a = "scheduler.process_batch_result_decode"
        long_b = "scheduler.process_batch_result_extend"
        assert name_skeleton(long_a) != name_skeleton(long_b)

    def test_cluster_and_dominant(self):
        names = ["step[DECODE bs=1]", "step[DECODE bs=2]", "step[EXTEND bs=1 toks=8]"]
        groups = cluster_by_skeleton(names)
        assert len(groups) == 2
        skeleton, share = dominant_cluster(groups)
        assert skeleton == "step[DECODE bs=#]"
        assert share == 2 / 3

    def test_dominant_of_nothing(self):
        assert dominant_cluster({}) == (None, 0.0)


# --------------------------------------------------------------------------- #
# C6: identity and inheritance
# --------------------------------------------------------------------------- #
class TestAnnotationIdentity:
    def test_memoized_instance_is_shared(self):
        assert parse_annotation("step[DECODE bs=4]") is parse_annotation(
            "step[DECODE bs=4]"
        )

    def test_parseable_reflects_recognition_not_the_detail_dict(self):
        assert is_parseable("step[DECODE bs=4]")
        assert not is_parseable("scheduler.process_batch_result")

    def test_unparseable_name_still_yields_a_full_detail_dict(self):
        """Why classification must not read the numbers: the stub looks real."""
        details = parse_annotation("scheduler.process_batch_result").iter_details()
        assert details["num_requests"] == 1
        assert details["context_requests"] == 0

    def test_resolution_exists_on_every_annotation(self):
        """Stage 2 keys on resolution; a missing attribute would raise."""
        assert parse_annotation("step[DECODE bs=4]").resolution is None


# --------------------------------------------------------------------------- #
# C2: periodicity
# --------------------------------------------------------------------------- #
class TestPeriodicity:
    def test_skips_a_warmup_prefix(self):
        period, pattern, start = _find_repeating_period(
            ["setup", "a", "b", "a", "b", "a", "b"]
        )
        assert (period, start) == (2, 1)
        assert pattern == ["a", "b"]

    def test_no_repetition_yields_nothing(self):
        assert _find_repeating_period(["a", "b", "c", "d"]) == (None, None, None)

    def test_too_few_repeats_rejected(self):
        assert _find_repeating_period(["a", "b", "a", "b"]) == (None, None, None)

    def test_reports_a_primitive_period_not_a_multiple(self):
        """Every multiple of a valid period is valid; only the unit is useful."""
        period, _, _ = _find_repeating_period(["a", "b"] * 12)
        assert period == 2

    def test_sub_iteration_noise_does_not_win(self):
        """A launch repeating within the iteration must not become the period."""
        labels = (["step"] + ["launch"] * 5) * 8
        period, _, _ = _find_repeating_period(labels)
        assert period == 6


# --------------------------------------------------------------------------- #
# C3: containment queries
# --------------------------------------------------------------------------- #
class TestIntervalIndex:
    def test_finds_events_inside_a_span_excluding_itself(self):
        outer = annotation("outer", 100, 100)
        inner = annotation("inner", 120, 20)
        index = IntervalIndex([outer, inner])
        assert index.contained_in(outer) == [inner]
        assert index.contained_in(outer, exclude_self=False) == [outer, inner]

    def test_partial_overlap_is_not_containment(self):
        outer = annotation("outer", 100, 100)
        straddling = annotation("straddling", 150, 100)
        index = IntervalIndex([outer, straddling])
        assert index.contained_in(outer) == []

    def test_containment_never_crosses_threads(self):
        outer = annotation("outer", 100, 100, tid=10)
        other_thread = annotation("inner", 120, 20, tid=11)
        index = IntervalIndex([outer, other_thread])
        assert index.contained_in(outer) == []

    def test_ignores_events_without_duration(self):
        outer = annotation("outer", 100, 100)
        flow = {"name": "ac2g", "ph": "s", "ts": 110, "pid": 1, "tid": 10}
        assert IntervalIndex([outer, flow]).contained_in(outer) == []

    def test_group_by_thread_sorts_within_group(self):
        events = [annotation("b", 200, 1, tid=10), annotation("a", 100, 1, tid=10)]
        groups = group_by_thread(events)
        assert [e["name"] for e in groups[(1, 10)]] == ["a", "b"]


# --------------------------------------------------------------------------- #
# C4: GPU attribution
# --------------------------------------------------------------------------- #
class TestGpuAttribution:
    def test_prefers_gpu_annotation_spans_when_present(self):
        events = serving_trace(4, with_gpu_annotation=True)
        attribution = GpuAttribution(TraceIndex(events))
        _, strategy = attribution.attributed_kernels(collect_annotations(events))
        assert strategy == GpuAttribution.STRATEGY_GPU_SPAN

    def test_falls_back_to_correlation_without_gpu_annotation_spans(self):
        events = serving_trace(4)
        attribution = GpuAttribution(TraceIndex(events))
        _, strategy = attribution.attributed_kernels(collect_annotations(events))
        assert strategy == GpuAttribution.STRATEGY_CORRELATION

    def test_strategy_is_chosen_per_root_set_not_per_trace(self):
        """One instance without a GPU counterpart must not measure by spans.

        A name join would credit this set with the annotated iterations' spans;
        the External id join sees the gap and falls back to correlation.
        """
        events = serving_trace(4, with_gpu_annotation=True)
        annotations = collect_annotations(events)
        unannotated = annotation("step[DECODE bs=9]", 90_000, 400)
        attribution = GpuAttribution(TraceIndex(events + [unannotated]))
        _, strategy = attribution.attributed_kernels(annotations + [unannotated])
        assert strategy == GpuAttribution.STRATEGY_CORRELATION

    def test_gpu_annotation_spans_are_excluded_from_gpu_busy_time(self):
        """Counting an annotation span as GPU time double-counts the kernels inside."""
        events = serving_trace(4, with_gpu_annotation=True)
        annotations = collect_annotations(events)
        assert GpuAttribution(TraceIndex(events)).audit(annotations).gpu_busy == 4 * 40

    def test_full_coverage_when_every_kernel_is_annotated(self):
        events = serving_trace(8)
        annotations = collect_annotations(events)
        report = GpuAttribution(TraceIndex(events)).audit(annotations)
        assert report.covered_selected == 1.0
        assert report.passes

    def test_work_just_outside_a_root_counts_once_windows_extend(self):
        """Extraction captures the tile, so the audit must judge the tile.

        Mirrors vLLM sampling: each iteration launches work just after its
        annotation ends. Judging bare spans would report a tenth of the GPU
        unaccounted for and send detection looking for extra roots, splitting
        every iteration in two to find work already being captured.
        """
        events, corr = [], 10
        for i in range(12):
            base = 1000 + i * 1000
            events.append(annotation(f"step[DECODE bs={i + 1}]", base, 400))
            events.append(launch(base + 10, corr))
            events.append(kernel(base + 100, 90, corr))
            corr += 1
            events.append(launch(base + 500, corr))  # after the annotation ends
            events.append(kernel(base + 600, 10, corr))
            corr += 1

        roots = collect_annotations(events)
        report = GpuAttribution(TraceIndex(events)).audit(roots)
        # A tenth of GPU time is launched after the annotations close.
        assert report.covered_spans == pytest.approx(0.9)
        # The windows reclaim it, bar the final iteration's tail, which falls
        # outside the last root and so outside the audited window.
        assert report.covered_selected > report.covered_spans
        assert report.passes

    def test_sparse_roots_stretched_over_many_iterations_do_not_pass(self):
        """Coverage from window extension alone is not root coverage."""
        events = serving_trace(40)
        annotations = collect_annotations(events)
        report = GpuAttribution(TraceIndex(events)).audit(annotations[::20])
        assert report.covered_selected > report.covered_spans
        assert report.span_share < 0.5
        assert not report.passes

    def test_gate_measures_the_roots_not_every_annotation(self):
        """The 0.5.17 lesson: blanket annotation coverage is not a root check.

        A run whose annotations cover the whole timeline while the chosen roots
        cover a fraction of its iterations must not pass.
        """
        events = serving_trace(40)
        annotations = collect_annotations(events)
        attribution = GpuAttribution(TraceIndex(events))
        assert attribution.audit(annotations).covered_selected == 1.0
        assert not attribution.audit(annotations[:2]).passes

    def test_unannotated_work_lowers_coverage(self):
        events = serving_trace(8)
        # A kernel with no launch site inside any annotation.
        events.append(kernel(1500, 4000, 99999, name="orphan"))
        attribution = GpuAttribution(TraceIndex(events))
        report = attribution.audit(collect_annotations(events))
        assert report.covered_selected < COVERAGE_GATE

    def test_selected_roots_can_cover_less_than_all_annotations(self):
        """The signature of roots sitting at the wrong nesting level."""
        events = serving_trace(8)
        annotations = collect_annotations(events)
        attribution = GpuAttribution(TraceIndex(events))
        assert (
            attribution.audit(annotations[:2]).covered_selected
            < attribution.audit(annotations).covered_selected
        )

    def test_family_gpu_time(self):
        events = serving_trace(4)
        attribution = GpuAttribution(TraceIndex(events))
        annotations = collect_annotations(events)
        assert attribution.gpu_time_for_family(annotations) == 4 * 40


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Stage 1 end to end
# --------------------------------------------------------------------------- #
class TestDetectionFlow:
    def test_healthy_trace_resolves_without_probes(self):
        result = _detect(serving_trace(16))
        assert result.status is DetectStatus.SPLITTABLE
        assert result.phase_confidence is PhaseConfidence.HIGH
        assert result.method == "annotation:tier"
        assert len(result) == 16
        assert result.coverage.covered_selected == 1.0

    def test_partial_known_falls_through_to_unknown_family(self):
        """Only 3 of 20 iterations match a known pattern. Known annotations
        have poor coverage, so the unknown family path picks up
        scheduler.run_batch which covers all 20.
        """
        events, corr = [], 400
        for i in range(20):
            base = 1000 + i * 1000
            events.append(annotation("scheduler.run_batch", base, 500))
            inner = (
                f"step[DECODE bs={i + 1}]" if i < 3 else f"step[TARGET_VERIFY bs={i}]"
            )
            events.append(annotation(inner, base + 50, 200))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1

        result = _detect(events)
        assert len(result) == 20
        assert result.method == "family:unknown_only"
        assert result.diagnostics["root_family"] == "scheduler.run_batch"
        assert result.status is DetectStatus.SPLITTABLE

    def test_unknown_family_with_nested_known_annotations(self):
        """When known annotations are nested inside an unknown family,
        the unknown family is selected because it covers all iterations.
        """
        events, corr = [], 600
        for i in range(12):
            base = 1000 + i * 1000
            events.append(annotation("scheduler.run_batch", base, 800))
            for step in range(2):
                inner = (
                    f"step[DECODE bs={step + 1}]" if i < 6 else f"step[UNKNOWN {step}]"
                )
                events.append(annotation(inner, base + 50 + step * 300, 200))
                events.append(launch(base + 60 + step * 300, corr))
                events.append(kernel(base + 900 + step * 50, 100, corr))
                corr += 1

        result = _detect(events)
        assert result.method == "family:unknown_only"
        assert len(result) == 12
        assert result.diagnostics["root_family"] == "scheduler.run_batch"

    def test_unknown_family_catches_enclosing_annotation(self):
        """When only a few iterations have known annotations, the unknown
        family path picks up the enclosing annotation that covers all."""
        events, corr = [], 900
        for i in range(20):
            base = 1000 + i * 1000
            events.append(annotation("scheduler.process_batch_result", base, 500))
            inner = f"step[DECODE bs={i + 1}]" if i < 3 else f"custom_step_{i}"
            events.append(annotation(inner, base + 50, 200))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1

        result = _detect(events)
        assert result.status is DetectStatus.SPLITTABLE
        assert result.method == "family:unknown_only"
        assert len(result) == 20
        assert (
            result.diagnostics["root_family"]
            == "scheduler.process_batch_result"
        )

    def test_known_annotations_used_when_all_match(self):
        """When every iteration has a known annotation, use them directly."""
        events, corr = [], 700
        for i in range(12):
            base = 1000 + i * 1000
            events.append(annotation(f"step[DECODE bs={i + 1}]", base + 50, 200))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1

        result = _detect(events)
        assert len(result) == 12
        assert result.method == "annotation:tier"

    def test_unrecognized_annotations_are_still_splittable(self):
        events, corr = [], 300
        for i in range(10):
            base = 1000 + i * 1000
            events.append(annotation("my_custom_step", base, 400))
            events.append(launch(base + 10, corr))
            events.append(kernel(base + 500, 200, corr))
            corr += 1

        result = _detect(events)
        assert len(result) == 10
        assert result.method == "family:unknown_only"
        assert result.phase_confidence is PhaseConfidence.UNKNOWN
        assert result.status is not DetectStatus.NOT_SPLITTABLE

    def test_empty_trace_reports_not_splittable(self):
        result = _detect([])
        assert result.status is DetectStatus.NOT_SPLITTABLE
        assert len(result) == 0

    def test_uncovered_work_grades_directly_without_probes(self):
        # Escalation probes were removed: uncovered work no longer triggers a
        # probe ladder; the roots are graded straight to degraded/not-splittable.
        events = serving_trace(16)
        events.append(kernel(1500, 500_000, 99999, name="unaccounted"))
        result = _detect(events)
        assert result.coverage.covered_selected < COVERAGE_GATE
        assert result.status in (DetectStatus.DEGRADED, DetectStatus.NOT_SPLITTABLE)

    def test_manifest_reports_quality(self):
        manifest = _detect(serving_trace(16)).to_manifest()
        assert manifest["status"] == 0
        assert manifest["phase_confidence"] == "high"
        assert manifest["n_roots"] == 16
        assert manifest["coverage_selected_roots"] == 1.0
        assert manifest["attribution_strategy"] == "correlation"


# --------------------------------------------------------------------------- #
# Stage 3: tiling
# --------------------------------------------------------------------------- #
class TestRootTiles:
    def test_windows_touch_so_gaps_belong_to_somebody(self):
        roots = [annotation("r", 1000, 100), annotation("r", 2000, 100)]
        tiles, overlaps = build_root_tiles(roots)
        assert overlaps == 0
        assert tiles[(1, 10, 1000)] == (1000, 2000)

    def test_last_window_uses_the_median_length(self):
        roots = [annotation("r", 1000, 100), annotation("r", 2000, 100)]
        tiles, _ = build_root_tiles(roots)
        assert tiles[(1, 10, 2000)] == (2000, 3000)

    def test_single_root_keeps_its_own_span(self):
        tiles, _ = build_root_tiles([annotation("r", 1000, 100)])
        assert tiles[(1, 10, 1000)] == (1000, 1100)

    def test_threads_are_tiled_independently(self):
        roots = [
            annotation("r", 1000, 100, tid=10),
            annotation("r", 1500, 100, tid=11),
            annotation("r", 3000, 100, tid=10),
        ]
        tiles, _ = build_root_tiles(roots)
        assert tiles[(1, 10, 1000)] == (1000, 3000)
        assert tiles[(1, 11, 1500)] == (1500, 1600)

    def test_overlapping_roots_keep_their_own_span_and_are_counted(self):
        roots = [annotation("r", 1000, 900), annotation("r", 1500, 100)]
        tiles, overlaps = build_root_tiles(roots)
        assert overlaps == 1
        assert tiles[(1, 10, 1000)] == (1000, 1900)


class TestGapFreeExtraction:
    def _trace(self):
        """Each iteration launches one kernel inside its root and one after it."""
        events, corr = [], 10
        for i in range(3):
            base = 1000 + i * 1000
            events.append(annotation(f"step[DECODE bs={i + 1}]", base, 100))
            events.append(launch(base + 10, corr))
            events.append(kernel(base + 300, 20, corr, name="k_in_root"))
            corr += 1
            events.append(launch(base + 400, corr))  # after the annotation ends
            events.append(kernel(base + 500, 30, corr, name="k_in_gap"))
            corr += 1
        return events

    def test_gap_kernels_are_recovered(self):
        events = self._trace()
        td = _trace_data(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        _, _, dropped, _, _ = extract_iteration(
            roots, td, gap_fill=False
        )
        out, _, kept, _, busy = extract_iteration(
            roots, td, root_tiles=tiles
        )
        assert dropped == 3
        assert kept == 6
        assert busy == 3 * 50
        names = {e["name"] for e in out["traceEvents"] if e.get("cat") == "kernel"}
        assert names == {"k_in_root", "k_in_gap"}

    def test_every_kernel_lands_in_exactly_one_window(self):
        events = self._trace()
        td = _trace_data(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        per_root = [
            extract_iteration(
                [r], td, root_tiles=tiles
            )[2]
            for r in roots
        ]
        total_in_trace = sum(1 for e in events if e.get("cat") == "kernel")
        assert sum(per_root) == total_in_trace

    def test_warmup_before_the_first_root_is_excluded(self):
        """Tiles span the iterations, not the capture.

        Work launched before the first root belongs to no iteration, so the
        per-iteration counts legitimately fall short of the trace total. Reading
        that shortfall as lost kernels reports healthy traces as broken; the
        failure worth detecting is a kernel claimed by two iterations.
        """
        events = self._trace()
        events.append(launch(500, 99))
        events.append(kernel(600, 40, 99, name="k_warmup"))
        td = _trace_data(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        per_root = [
            extract_iteration(
                [r], td, root_tiles=tiles
            )[2]
            for r in roots
        ]
        total_in_trace = sum(1 for e in events if e.get("cat") == "kernel")
        assert sum(per_root) == total_in_trace - 1
        assert sum(per_root) <= total_in_trace

    def test_enclosing_spans_are_left_out(self):
        """An outer frame belongs to no single iteration."""
        events = self._trace()
        events.append(
            {
                "name": "whole_run",
                "cat": "python_function",
                "ph": "X",
                "ts": 900,
                "dur": 5000,
                "pid": 1,
                "tid": 10,
                "args": {},
            }
        )
        td = _trace_data(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)
        out, _, _, _, _ = extract_iteration(
            [roots[0]], td, root_tiles=tiles
        )
        assert "whole_run" not in {e["name"] for e in out["traceEvents"]}
