###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the coverage-gated splitter: components and the flow built on them."""

from TraceLens.Trace2Tree.inference_iteration_roots import (
    PERIOD_CONFLICT,
    PERIOD_EXACT,
    PERIOD_INTEGER_RATIO,
    compare_periods,
    find_period_candidates,
)
from TraceLens.TraceUtils.annotation_utils import (
    PROVENANCE_KEY,
    cluster_by_skeleton,
    dominant_cluster,
    inherit_identity,
    is_parseable,
    name_skeleton,
    parse_annotation,
)
from TraceLens.TraceUtils.split_inference import (
    DetectStatus,
    PhaseConfidence,
    build_root_tiles,
    classify_workload,
    extract_iteration,
    find_iteration_roots_ex,
    find_max_pattern_window,
    preprocess_trace,
    select_window,
)
from TraceLens.TraceUtils.split_inference.detect_utils import (
    COVERAGE_GATE,
    GpuAttribution,
    IntervalIndex,
    gaps_between,
    group_by_thread,
)
from TraceLens.TraceUtils.split_inference.root_detection import (
    build_families,
    collect_annotations,
    resolve_nesting,
)

VLLM = "execute_{i}_context_3(sq128sk256sqsq1sqsk1)_generation_2(sq1sk300sqsq1sqsk1)"


# --------------------------------------------------------------------------- #
# Event builders
# --------------------------------------------------------------------------- #
def annotation(name, ts, dur, pid=1, tid=10):
    return {
        "name": name,
        "cat": "user_annotation",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {},
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


def projection(name, ts, dur, pid=1, tid=99):
    return {
        "name": name,
        "cat": "gpu_user_annotation",
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": {},
    }


def serving_trace(count=16, name_template=VLLM, period=1000, with_projection=False):
    """One annotation per iteration, each launching one kernel."""
    events, corr = [], 500
    for i in range(count):
        base = 1000 + i * period
        name = name_template.format(i=i)
        events.append(annotation(name, base, 100))
        events.append(launch(base + 10, corr))
        events.append(kernel(base + 200, 40, corr))
        if with_projection:
            events.append(projection(name, base + 200, 40))
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

    def test_inherit_keeps_window_and_takes_identity(self):
        outer = annotation("scheduler.process_batch_result", 1000, 500)
        inner = annotation("step[DECODE bs=7]", 1050, 100)
        merged = inherit_identity(outer, inner)

        assert merged["ts"] == 1000 and merged["dur"] == 500
        assert merged["name"] == "step[DECODE bs=7]"
        assert parse_annotation(merged["name"]).generation_requests == 7
        assert merged[PROVENANCE_KEY] == {
            "window_from": "scheduler.process_batch_result",
            "identity_from": "step[DECODE bs=7]",
        }
        assert outer["name"] == "scheduler.process_batch_result"

    def test_second_inheritance_keeps_the_original_window_owner(self):
        outer = annotation("scheduler.process_batch_result", 1000, 500)
        once = inherit_identity(outer, annotation("step[DECODE bs=1]", 1050, 100))
        twice = inherit_identity(once, annotation("step[EXTEND bs=2 toks=8]", 1060, 10))
        assert twice[PROVENANCE_KEY]["window_from"] == "scheduler.process_batch_result"
        assert twice[PROVENANCE_KEY]["identity_from"] == "step[EXTEND bs=2 toks=8]"


# --------------------------------------------------------------------------- #
# C2: periodicity
# --------------------------------------------------------------------------- #
class TestPeriodicity:
    def test_skips_a_warmup_prefix(self):
        best = find_period_candidates(["setup", "a", "b", "a", "b", "a", "b"])[0]
        assert (best.period, best.start, best.repeats) == (2, 1, 3)

    def test_no_repetition_yields_nothing(self):
        assert find_period_candidates(["a", "b", "c", "d"]) == []

    def test_too_few_repeats_rejected(self):
        assert find_period_candidates(["a", "b", "a", "b"]) == []

    def test_reports_a_primitive_period_not_a_multiple(self):
        """Every multiple of a valid period is valid; only the unit is useful."""
        candidates = find_period_candidates(["a", "b"] * 12)
        assert candidates[0].period == 2
        assert all(c.period % 2 or c.period == 2 for c in candidates)

    def test_sub_iteration_noise_does_not_win(self):
        """A launch repeating within the iteration must not become the period."""
        labels = (["step"] + ["launch"] * 5) * 8
        best = find_period_candidates(labels)[0]
        assert best.period == 6
        assert best.repeats == 8

    def test_duration_variance_is_reported(self):
        labels = ["a", "b"] * 5
        steady = find_period_candidates(labels, durations=[10, 20] * 5)[0]
        erratic = find_period_candidates(
            labels, durations=[10, 20, 900, 20, 10, 20, 10, 500, 10, 20]
        )[0]
        assert steady.duration_cv == 0.0
        assert erratic.duration_cv > steady.duration_cv

    def test_coverage_reported(self):
        best = find_period_candidates(["x", "y"] * 10)[0]
        assert best.coverage == 1.0

    def test_compare_periods(self):
        assert compare_periods(4, 4) == (PERIOD_EXACT, 1)
        assert compare_periods(4, 12) == (PERIOD_INTEGER_RATIO, 3)
        assert compare_periods(4, 7) == (PERIOD_CONFLICT, None)
        assert compare_periods(4, None) == (PERIOD_CONFLICT, None)
        assert compare_periods(0, 4) == (PERIOD_CONFLICT, None)


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

    def test_gaps_between_consecutive_roots(self):
        roots = [annotation("r", 100, 10), annotation("r", 200, 10)]
        (gap,) = gaps_between(roots)
        assert (gap["ts"], gap["dur"]) == (110, 90)

    def test_no_gap_when_roots_touch(self):
        roots = [annotation("r", 100, 100), annotation("r", 200, 10)]
        assert gaps_between(roots) == []

    def test_group_by_thread_sorts_within_group(self):
        events = [annotation("b", 200, 1, tid=10), annotation("a", 100, 1, tid=10)]
        groups = group_by_thread(events)
        assert [e["name"] for e in groups[(1, 10)]] == ["a", "b"]


# --------------------------------------------------------------------------- #
# C4: GPU attribution
# --------------------------------------------------------------------------- #
class TestGpuAttribution:
    def test_prefers_projections_when_present(self):
        attribution = GpuAttribution(serving_trace(4, with_projection=True))
        assert attribution.strategy == GpuAttribution.STRATEGY_PROJECTION

    def test_falls_back_to_correlation_without_projections(self):
        attribution = GpuAttribution(serving_trace(4))
        assert attribution.strategy == GpuAttribution.STRATEGY_CORRELATION

    def test_projections_are_excluded_from_gpu_busy_time(self):
        """Counting a projection as GPU time double-counts the kernels inside it."""
        events = serving_trace(4, with_projection=True)
        annotations = collect_annotations(events)
        assert GpuAttribution(events).audit(annotations, annotations).gpu_busy == 4 * 40

    def test_full_coverage_when_every_kernel_is_annotated(self):
        events = serving_trace(8)
        annotations = collect_annotations(events)
        report = GpuAttribution(events).audit(annotations, annotations)
        assert report.covered_any == 1.0
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
        report = GpuAttribution(events).audit(roots, roots)
        assert report.covered_spans < 1.0
        assert report.covered_selected == 1.0
        assert report.span_share > 0.9
        assert report.passes

    def test_sparse_roots_stretched_over_many_iterations_do_not_pass(self):
        """Coverage from window extension alone is not root coverage."""
        events = serving_trace(40)
        annotations = collect_annotations(events)
        report = GpuAttribution(events).audit(annotations, annotations[::20])
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
        report = GpuAttribution(events).audit(annotations, annotations[:2])
        assert report.covered_any == 1.0
        assert not report.passes
        assert report.better_roots_exist

    def test_unannotated_work_lowers_coverage(self):
        events = serving_trace(8)
        # A kernel with no launch site inside any annotation.
        events.append(kernel(1500, 4000, 99999, name="orphan"))
        attribution = GpuAttribution(events)
        report = attribution.audit(collect_annotations(events), [])
        assert report.covered_any < COVERAGE_GATE
        assert [
            k["name"]
            for k in attribution.uncovered_kernels(collect_annotations(events))
        ] == ["orphan"]

    def test_selected_roots_can_cover_less_than_all_annotations(self):
        """The signature of roots sitting at the wrong nesting level."""
        events = serving_trace(8)
        annotations = collect_annotations(events)
        attribution = GpuAttribution(events)
        report = attribution.audit(annotations, annotations[:2])
        assert report.covered_any == 1.0
        assert report.covered_selected < report.covered_any

    def test_family_gpu_time_and_launch_sites(self):
        events = serving_trace(4)
        attribution = GpuAttribution(events)
        annotations = collect_annotations(events)
        skeleton = name_skeleton(annotations[0]["name"])
        assert attribution.gpu_time_for_family(skeleton, annotations) == 4 * 40
        assert len(attribution.cpu_launches_for(attribution.kernels)) == 4

    def test_graph_launched_kernels_have_no_launch_site(self):
        attribution = GpuAttribution([kernel(100, 10, 4242)])
        assert attribution.cpu_launches_for(attribution.kernels) == []


# --------------------------------------------------------------------------- #
# C9: families
# --------------------------------------------------------------------------- #
class TestFamilies:
    def _events(self):
        """A scheduler family wrapping a decode family, plus CPU-only chatter."""
        events = []
        corr = 800
        for i in range(12):
            base = 1000 + i * 1000
            events.append(annotation("scheduler.process_batch_result", base, 500))
            events.append(annotation(f"step[DECODE bs={i + 1}]", base + 50, 200))
            events.append(annotation("scheduler.log_stats", base + 700, 10))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1
        return events

    def test_prunes_families_with_no_gpu_work(self):
        events = self._events()
        families = build_families(collect_annotations(events), GpuAttribution(events))
        skeletons = {f.skeleton for f in families}
        assert "scheduler.log_stats" not in skeletons
        assert {"scheduler.process_batch_result", "step[DECODE bs=#]"} <= skeletons

    def test_nesting_is_directional(self):
        events = self._events()
        annotations = collect_annotations(events)
        families = build_families(annotations, GpuAttribution(events))
        index = IntervalIndex(annotations)
        resolve_nesting(families, index)
        by_skeleton = {f.skeleton: f for f in families}
        outer = by_skeleton["scheduler.process_batch_result"]
        inner = by_skeleton["step[DECODE bs=#]"]
        assert outer.encloses["step[DECODE bs=#]"] == 12
        assert inner.encloses["scheduler.process_batch_result"] == 0

    def test_regularity_needs_enough_instances(self):
        events = serving_trace(3)
        families = build_families(collect_annotations(events), GpuAttribution(events))
        assert families and not families[0].regular

    def test_parseability_recorded_per_family(self):
        events = self._events()
        families = build_families(collect_annotations(events), GpuAttribution(events))
        by_skeleton = {f.skeleton: f for f in families}
        assert by_skeleton["step[DECODE bs=#]"].parseable
        assert not by_skeleton["scheduler.process_batch_result"].parseable


# --------------------------------------------------------------------------- #
# Stage 1 end to end
# --------------------------------------------------------------------------- #
class TestDetectionFlow:
    def test_healthy_trace_resolves_without_probes(self):
        result = find_iteration_roots_ex(serving_trace(16))
        assert result.status is DetectStatus.SPLITTABLE
        assert result.phase_confidence is PhaseConfidence.HIGH
        assert result.method == "annotation:tier"
        assert len(result) == 16
        assert result.coverage.covered_any == 1.0
        assert result.diagnostics["probes_run"] == []

    def test_whole_outer_family_is_adopted_not_just_matching_instances(self):
        """The 0.5.17 shape: most iterations wrap an unrecognized annotation.

        Only the first three iterations carry a name a parser knows. Keeping just
        those would split a twenty-iteration run into three.
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

        result = find_iteration_roots_ex(events)
        assert len(result) == 20
        assert result.method == "annotation:widened"
        assert result.diagnostics["root_family_skeleton"] == "scheduler.run_batch"
        # Three roots parsed, seventeen did not, so the phases are not all real.
        assert result.diagnostics["n_roots_with_phase"] == 3
        assert result.phase_confidence is PhaseConfidence.LOW
        assert result.status is DetectStatus.SPLITTABLE

    def test_outer_family_found_when_it_wraps_several_roots_each(self):
        """One outer span per two known roots still counts as wrapping them.

        Counting enclosing instances rather than enclosed roots reports half here
        and abandons the widening, keeping the two roots instead of all twelve.
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

        result = find_iteration_roots_ex(events)
        assert result.method == "annotation:widened"
        assert len(result) == 12
        assert result.diagnostics["root_family_skeleton"] == "scheduler.run_batch"

    def test_unknown_outer_family_becomes_the_window(self):
        """The useful span has a name no regex knows."""
        events, corr = [], 900
        for i in range(20):
            base = 1000 + i * 1000
            events.append(annotation("scheduler.process_batch_result", base, 500))
            events.append(annotation(f"step[DECODE bs={i + 1}]", base + 50, 200))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1

        result = find_iteration_roots_ex(events)
        assert result.status is DetectStatus.SPLITTABLE
        assert result.method == "annotation:widened"
        assert len(result) == 20
        # Window from the scheduler span, identity from the decode annotation.
        assert result.roots[0]["dur"] == 500
        assert result.roots[0][PROVENANCE_KEY] == {
            "window_from": "scheduler.process_batch_result",
            "identity_from": "step[DECODE bs=1]",
        }
        assert result.phase_confidence is PhaseConfidence.HIGH
        assert result.diagnostics["root_family_known"] is False

    def test_inner_annotation_relabels_a_known_root(self):
        """The 0.5.11 shape: the outer name parses, the inner one is truer."""
        events, corr = [], 700
        for i in range(12):
            base = 1000 + i * 1000
            events.append(annotation(VLLM.format(i=i), base, 500))
            events.append(annotation(f"step[DECODE bs={i + 1}]", base + 50, 200))
            events.append(launch(base + 60, corr))
            events.append(kernel(base + 600, 300, corr))
            corr += 1

        result = find_iteration_roots_ex(events)
        assert len(result) == 12
        assert result.roots[0]["dur"] == 500
        assert result.roots[0][PROVENANCE_KEY]["identity_from"] == "step[DECODE bs=1]"

    def test_unrecognized_annotations_are_still_splittable(self):
        events, corr = [], 300
        for i in range(10):
            base = 1000 + i * 1000
            events.append(annotation("my_custom_step", base, 400))
            events.append(launch(base + 10, corr))
            events.append(kernel(base + 500, 200, corr))
            corr += 1

        result = find_iteration_roots_ex(events)
        assert len(result) == 10
        assert result.method == "family:unknown_only"
        assert result.phase_confidence is PhaseConfidence.UNKNOWN
        assert result.status is not DetectStatus.NOT_SPLITTABLE

    def test_empty_trace_reports_not_splittable(self):
        result = find_iteration_roots_ex([])
        assert result.status is DetectStatus.NOT_SPLITTABLE
        assert len(result) == 0

    def test_uncovered_work_triggers_probes_and_is_recorded(self):
        events = serving_trace(16)
        events.append(kernel(1500, 500_000, 99999, name="unaccounted"))
        result = find_iteration_roots_ex(events)
        assert result.diagnostics["probes_run"], "escalation must be recorded"
        assert result.coverage.covered_any < COVERAGE_GATE
        assert result.status in (DetectStatus.DEGRADED, DetectStatus.NOT_SPLITTABLE)

    def test_manifest_reports_quality(self):
        manifest = find_iteration_roots_ex(serving_trace(16)).to_manifest()
        assert manifest["status"] == 0
        assert manifest["phase_confidence"] == "high"
        assert manifest["n_roots"] == 16
        assert manifest["coverage_any_annotation"] == 1.0
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
        trace = {"traceEvents": events}
        gpu_map, flow_map, meta = preprocess_trace(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        _, _, dropped, _, _ = extract_iteration(
            roots, events, trace, gpu_map, flow_map, meta, gap_fill=False
        )
        out, _, kept, _, busy = extract_iteration(
            roots, events, trace, gpu_map, flow_map, meta, root_tiles=tiles
        )
        assert dropped == 3
        assert kept == 6
        assert busy == 3 * 50
        names = {e["name"] for e in out["traceEvents"] if e.get("cat") == "kernel"}
        assert names == {"k_in_root", "k_in_gap"}

    def test_every_kernel_lands_in_exactly_one_window(self):
        events = self._trace()
        trace = {"traceEvents": events}
        gpu_map, flow_map, meta = preprocess_trace(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        per_root = [
            extract_iteration(
                [r], events, trace, gpu_map, flow_map, meta, root_tiles=tiles
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
        trace = {"traceEvents": events}
        gpu_map, flow_map, meta = preprocess_trace(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)

        per_root = [
            extract_iteration(
                [r], events, trace, gpu_map, flow_map, meta, root_tiles=tiles
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
        trace = {"traceEvents": events}
        gpu_map, flow_map, meta = preprocess_trace(events)
        roots = collect_annotations(events)
        tiles, _ = build_root_tiles(roots)
        out, _, _, _, _ = extract_iteration(
            [roots[0]], events, trace, gpu_map, flow_map, meta, root_tiles=tiles
        )
        assert "whole_run" not in {e["name"] for e in out["traceEvents"]}


# --------------------------------------------------------------------------- #
# Stage 2: workload classification and window choice
# --------------------------------------------------------------------------- #
class TestWorkloadClassification:
    def test_serving_recognized(self):
        roots = collect_annotations(serving_trace(8))
        workload, info = classify_workload(roots)
        assert workload == "serving"
        assert info["n_recognized_roots"] == 8

    def test_unrecognized_names_are_generic_not_decode_only(self):
        """Classification must not be fooled by the fabricated detail dict."""
        roots = [annotation("denoise_step", 1000 + i * 100, 50) for i in range(8)]
        workload, info = classify_workload(roots)
        assert workload == "generic"
        assert info["n_recognized_roots"] == 0
        assert info["annotation_kinds"] == {}

    def test_a_recognized_minority_does_not_make_it_serving(self):
        """Most steps unparsed means most request counts are invented."""
        roots = [
            annotation("scheduler.run_batch", 1000 + i * 100, 50) for i in range(40)
        ]
        roots += [
            annotation(f"step[DECODE bs={i}]", 9000 + i * 100, 50) for i in range(3)
        ]
        workload, info = classify_workload(roots)
        assert workload == "generic"
        assert info["n_recognized_roots"] == 3

    def test_pattern_window_skips_erratic_warmup(self):
        """Same shape throughout, so duration steadiness decides."""
        roots = []
        for i in range(4):  # warmup: same name, wildly uneven durations
            roots.append(annotation("denoise_step", 1000 + i * 5000, 4000 - i * 900))
        for i in range(8):  # steady state
            roots.append(annotation("denoise_step", 30000 + i * 1000, 900))

        window = find_max_pattern_window(roots, num_steps=4)
        assert len(window) == 4
        assert all(r["ts"] >= 30000 for r in window)

    def test_pattern_window_prefers_the_dominant_shape(self):
        roots = [annotation("odd_step", 1000 + i * 100, 50) for i in range(3)]
        roots += [annotation("denoise_step", 2000 + i * 100, 50) for i in range(9)]
        window = find_max_pattern_window(roots, num_steps=4)
        assert {r["name"] for r in window} == {"denoise_step"}

    def test_pattern_window_handles_a_short_run(self):
        roots = [annotation("denoise_step", 1000 + i * 100, 50) for i in range(2)]
        assert len(find_max_pattern_window(roots, num_steps=8)) == 2

    def test_pattern_window_of_nothing(self):
        assert find_max_pattern_window([], num_steps=4) == []

    def test_dispatch_records_the_strategy_used(self):
        serving = collect_annotations(serving_trace(24))
        _, info = select_window(serving, num_steps=4, steady_state_regions=[(0, 24)])
        assert info["window_strategy"].startswith("steady_state:")

        generic = [annotation("denoise_step", 1000 + i * 100, 50) for i in range(12)]
        window, info = select_window(generic, num_steps=4)
        assert info["window_strategy"] == "max_pattern_coverage"
        assert info["n_window_roots"] == len(window) == 4
