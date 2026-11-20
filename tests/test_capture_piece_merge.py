###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for SGLang spec-decode capture matching (verify, prefill, draft prefill)."""

from TraceLens.TraceUtils.utils.annotation_utils import (
    IterationAnnotation,
    spec_decode_role,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    UID,
    _batch_sizes_by_mode,
    _draft_prefill_tokens,
    _preferred_target_mode,
    _step_label,
    innermost_extend_steps,
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
    assert _preferred_target_mode("scheduler.run_batch") is None
    assert _preferred_target_mode(None) is None


def test_draft_prefill_tokens():
    assert _draft_prefill_tokens("step[DRAFT bs=1 toks=2048 c_sq=2048]") == 2048
    assert _draft_prefill_tokens("step[VERIFY bs=8]") is None
    assert _draft_prefill_tokens(None) is None


def _span(uid, name, ts, dur):
    return {UID: uid, "name": name, "ts": ts, "dur": dur, "cat": "user_annotation"}


def test_innermost_extend_drops_outer_spec_prefill_wrapper():
    outer = _span(1, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 0, 1000)
    inner = _span(2, "step[EXTEND bs=1 toks=2048 c_sq=2048 c_sqsq=1]", 10, 800)
    plain = _span(3, "step[EXTEND bs=1 toks=512 c_sq=512 c_sqsq=1]", 2000, 300)
    assert [e[UID] for e in innermost_extend_steps([outer, inner, plain])] == [2, 3]


def test_step_label_finds_draft_label_inside_draft_extend():
    draft_extend = {"name": "draft_extend", "ts": 100, "dur": 50}
    steps = [
        _span(1, "step[EXTEND bs=1 toks=2048]", 0, 200),
        _span(2, "step[DRAFT bs=1 toks=2048]", 110, 30),
    ]
    assert _step_label(steps, draft_extend) == "step[DRAFT bs=1 toks=2048]"
    assert _step_label(steps, {"name": "step[VERIFY bs=8]"}) == "step[VERIFY bs=8]"


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
