###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for SGLang spec-decode capture matching (verify, prefill, draft prefill)."""

from types import SimpleNamespace

from TraceLens.TraceUtils.utils.annotation_utils import (
    IterationAnnotation,
    find_step_annotations,
    find_step_label,
    innermost_extend_steps,
    parse_draft_prefill_tokens,
    spec_decode_role,
)
import TraceLens.Trace2Tree.trace_capture_merge_experimental as merge
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    UID,
    _batch_sizes_by_mode,
    _candidate_capture_sizes,
    _merge_capture_pair,
    _pair_capture_roots,
    _preferred_target_mode,
    _resolve_capture_query,
    find_execution_roots,
)


def test_sglang_verify_with_forward_id_parses_requests():
    ann = IterationAnnotation("step[VERIFY bs=7 forward_id=866]")
    assert ann.matched
    assert ann.generation_requests == 7
    assert ann.kind == "sglang_native"
    assert spec_decode_role("step[VERIFY bs=7 forward_id=866]") == "target"


def test_sglang_extend_reads_toks_after_forward_id():
    ann = IterationAnnotation("step[EXTEND bs=1 forward_id=3 toks=2048]")
    assert ann.matched
    assert ann.context_requests == 1
    assert ann.context_sum == 2048
    assert ann.batch_size == 2048


def test_sglang_plain_labels_unchanged():
    ann = IterationAnnotation("step[DECODE bs=64]")
    assert ann.matched
    assert ann.generation_requests == 64


def test_preferred_target_mode():
    assert _preferred_target_mode("step[EXTEND bs=1 toks=2048]") == "PIECEWISE"
    assert _preferred_target_mode("step[VERIFY bs=8 forward_id=1]") == "FULL"
    assert _preferred_target_mode("step[DECODE bs=8]") == "FULL"
    assert _preferred_target_mode("step[MIXED bs=2]") is None
    assert _preferred_target_mode("scheduler.run_batch") is None
    assert _preferred_target_mode(None) is None


def test_draft_prefill_tokens():
    assert parse_draft_prefill_tokens("step[DRAFT bs=1 toks=2048 c_sq=2048]") == 2048
    assert parse_draft_prefill_tokens("step[VERIFY bs=8]") is None
    assert parse_draft_prefill_tokens(None) is None


def _span(uid, name, ts, dur):
    return {UID: uid, "name": name, "ts": ts, "dur": dur, "cat": "user_annotation"}


def test_innermost_extend_drops_outer_spec_prefill_wrapper():
    outer = _span(1, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 0, 1000)
    inner = _span(2, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 10, 800)
    plain = _span(3, "step[EXTEND bs=1 toks=512 c_sq=512 c_sqsq=1]", 2000, 300)
    assert [e[UID] for e in innermost_extend_steps([outer, inner, plain])] == [2, 3]


def test_innermost_extend_ignores_gpu_annotation_copies():
    outer = _span(1, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 0, 1000)
    inner = _span(2, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 10, 800)
    gpu_outer = dict(outer, **{UID: 11, "cat": "gpu_user_annotation", "ts": 5})
    gpu_inner = dict(inner, **{UID: 12, "cat": "gpu_user_annotation", "ts": 15})
    events = [outer, gpu_outer, inner, gpu_inner]
    assert [e[UID] for e in innermost_extend_steps(events)] == [2]
    roots = find_execution_roots(SimpleNamespace(events=events))
    assert [e[UID] for e in roots] == [2]


def test_step_annotations_keep_only_cpu_labels():
    cpu = _span(1, "step[VERIFY bs=8 forward_id=4]", 0, 100)
    gpu = dict(cpu, **{UID: 2, "cat": "gpu_user_annotation"})
    other = _span(3, "scheduler.run_batch", 0, 100)
    assert [e[UID] for e in find_step_annotations([gpu, cpu, other])] == [1]


def test_step_label_finds_draft_label_inside_draft_extend():
    draft_extend = {"name": "draft_extend", "ts": 100, "dur": 50}
    steps = [
        _span(1, "step[EXTEND bs=1 toks=2048]", 0, 200),
        _span(2, "step[DRAFT bs=1 toks=2048]", 110, 30),
    ]
    assert find_step_label(steps, draft_extend) == "step[DRAFT bs=1 toks=2048]"
    assert find_step_label(steps, {"name": "step[VERIFY bs=8]"}) == "step[VERIFY bs=8]"
    assert find_step_label(steps, {"name": "draft", "ts": 500, "dur": 10}) is None


def test_batch_sizes_split_by_mode():
    capture_map = {
        "target_28_PIECEWISE": ["a"],
        "target_2048_PIECEWISE": ["b"],
        "target_32_FULL": ["c"],
        "drafter_extend_32_FULL": ["d"],
    }
    sizes = _batch_sizes_by_mode(capture_map)
    assert sorted(sizes[("target", "PIECEWISE")]) == [28, 2048]
    assert sizes[("target", "FULL")] == [32]
    assert sizes[("drafter_extend", "FULL")] == [32]


def test_find_execution_roots_uses_innermost_extend():
    outer = _span(1, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 0, 1000)
    inner = _span(2, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 10, 800)
    verify = _span(3, "step[VERIFY bs=8 forward_id=4]", 2000, 100)
    roots = find_execution_roots(SimpleNamespace(events=[outer, inner, verify]))
    assert [e[UID] for e in roots] == [2, 3]


def test_resolve_capture_query_draft_prefill():
    root = {"name": "draft_extend"}
    assert _resolve_capture_query(root, "step[DRAFT bs=1 toks=2048]", None) == (
        "drafter",
        (2048,),
        "PIECEWISE",
    )


def test_resolve_capture_query_target_and_drafter():
    verify = {"name": "step[VERIFY bs=7 forward_id=1]"}
    assert _resolve_capture_query(verify, verify["name"], None) == (
        "target",
        ("7",),
        "FULL",
    )
    # draft_extend outside a prefill keeps its own role and spec batch size.
    assert _resolve_capture_query(
        {"name": "draft_extend"}, "step[VERIFY bs=8]", (8,)
    ) == ("drafter_extend", (8,), None)
    assert _resolve_capture_query({"name": "draft"}, None, None) == (
        "drafter",
        None,
        None,
    )


def test_candidate_capture_sizes_rounds_up_within_mode():
    mult = {"target": 4}
    # bs-7 verify pads to the bs-32 decode graph (7 * 4 = 28 -> 32).
    assert _candidate_capture_sizes((7,), "target", "FULL", [32], mult) == [32]
    # Prefill tokens are not scaled by the request-bs multiplier.
    assert _candidate_capture_sizes(
        (2048,), "target", "PIECEWISE", [28, 2048], mult
    ) == [2048]
    assert _candidate_capture_sizes((64,), "target", "FULL", [32], mult) == []


def _source(mode, n_roots, tag):
    roots = [{"name": f"{tag}{i}"} for i in range(n_roots)]
    data = [([], set()) for _ in range(n_roots)]
    return (tag, roots, data, mode)


def test_pair_capture_roots_prefers_mode_and_pairs_positionally():
    graph_roots = [{"name": "g0"}, {"name": "g1"}]
    sources = [_source("PIECEWISE", 1, "pw"), _source("FULL", 2, "full")]
    pairs, data, trees = _pair_capture_roots(None, graph_roots, sources, "FULL")
    assert [(c["name"], g["name"]) for c, g in pairs] == [
        ("full0", "g0"),
        ("full1", "g1"),
    ]
    assert trees == ["full", "full"]

    pairs, _, trees = _pair_capture_roots(None, graph_roots, sources, None)
    assert [c["name"] for c, _ in pairs] == ["pw0", "pw0"]
    assert trees == ["pw", "pw"]


def test_pair_capture_roots_falls_back_to_count_matching():
    assert _pair_capture_roots(None, [{"name": "g0"}], [], "FULL") is None


def _stub_graph_kernels(monkeypatch, kernels):
    monkeypatch.setattr(merge, "get_subtree_events", lambda *a, **k: ([], kernels))


def _merge_pair():
    graph_tree = SimpleNamespace(events=[{UID: 0}])
    c_root, g_root = {"name": "CaptureRoot"}, {"name": "hipGraphLaunch"}
    return _merge_capture_pair(graph_tree, c_root, g_root, ([], set()), None)


def test_merge_capture_pair_skips_graph_without_kernels(monkeypatch):
    _stub_graph_kernels(monkeypatch, [])
    assert _merge_pair()[1] is False


def test_merge_capture_pair_multistream_without_names_fails(monkeypatch):
    _stub_graph_kernels(monkeypatch, [{"name": "k"}])
    monkeypatch.setattr(merge, "is_multistream", lambda events: True)
    monkeypatch.setattr(merge, "capture_has_kernel_names", lambda events: False)
    assert _merge_pair()[1] is True


def test_merge_capture_pair_skips_failed_stream_alignment(monkeypatch):
    _stub_graph_kernels(monkeypatch, [{"name": "k"}])
    monkeypatch.setattr(merge, "is_multistream", lambda events: True)
    monkeypatch.setattr(merge, "capture_has_kernel_names", lambda events: True)
    monkeypatch.setattr(merge, "align_streams", lambda g, c: None)
    assert _merge_pair()[1] is False


def test_merge_capture_pair_skips_failed_verification(monkeypatch):
    _stub_graph_kernels(monkeypatch, [{"name": "k"}])
    monkeypatch.setattr(merge, "is_multistream", lambda events: False)
    monkeypatch.setattr(merge, "verify_subtree_events", lambda c, g: (0, c, g))
    assert _merge_pair()[1] is False


def test_merge_capture_pair_grafts_aligned_subtree(monkeypatch):
    _stub_graph_kernels(monkeypatch, [{"name": "k"}])
    monkeypatch.setattr(merge, "is_multistream", lambda events: True)
    monkeypatch.setattr(merge, "capture_has_kernel_names", lambda events: True)
    monkeypatch.setattr(merge, "align_streams", lambda g, c: c)
    grafted = []
    monkeypatch.setattr(
        merge, "_graft_capture_subtree", lambda tree, *a: grafted.append(a) or tree
    )
    _, failed = _merge_pair()
    assert failed is False
    assert len(grafted) == 1


def test_merge_capture_trace_into_graph_skips_and_aborts(monkeypatch):
    def root(name):
        return {"name": name, "ts": 0, "dur": 1}

    graph = {"name": "hipGraphLaunch"}
    root_map = [
        (root("step[VERIFY bs=8 forward_id=1]"), []),  # no graph roots
        (root("scheduler.run_batch"), [graph]),  # unknown batch size
        (root("step[VERIFY bs=64 forward_id=2]"), [graph]),  # no capture size
        (root("step[VERIFY bs=8 forward_id=3]"), [graph]),  # no capture match
        (root("step[VERIFY bs=8 forward_id=4]"), [graph]),  # merged, then fails
    ]
    tree = SimpleNamespace(events=[], build_tree=lambda **k: None)
    monkeypatch.setattr(merge, "build_execution_graph_root_map", lambda t: root_map)
    monkeypatch.setattr(
        merge,
        "load_capture_folder",
        lambda *a: ({"target_32_FULL": ["f"]}, {"target": [32]}, {"target": 4}),
    )
    monkeypatch.setattr(
        merge, "spec_decode_batch_sizes", lambda roots: [None] * len(roots)
    )
    sources = iter([[], [_source("FULL", 1, "full")]])
    monkeypatch.setattr(merge, "_load_all_capture_sources", lambda *a: next(sources))
    monkeypatch.setattr(merge, "_merge_capture_pair", lambda t, *a: (t, True))

    result = merge.merge_capture_trace_into_graph(
        capture_folder="caps",
        metadata_json_path="caps/execution_details.json",
        graph_tree_filepath="replay-TP-0-EP-0.trace.json.gz",
        graph_tree=tree,
    )
    assert result is None
