###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for semantic_analyses/kernel_unification.py pure transforms."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEM = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis", "semantic_analyses")
sys.path.insert(0, SEM)

import kernel_unification  # noqa: E402

# ---------------------------------------------------------------------------
# _load / _dims_repr
# ---------------------------------------------------------------------------


def test_load_reads_json(tmp_path):
    p = tmp_path / "doc.json"
    p.write_text(json.dumps({"a": 1, "b": [2, 3]}))
    assert kernel_unification._load(str(p)) == {"a": 1, "b": [2, 3]}


def test_dims_repr_empty():
    assert kernel_unification._dims_repr(None) == ""
    assert kernel_unification._dims_repr([]) == ""


def test_dims_repr_compact():
    assert kernel_unification._dims_repr([[1, 2], [3, 4]]) == "[[1,2],[3,4]]"


def test_dims_repr_truncates():
    dims = [[i, i + 1] for i in range(200)]
    out = kernel_unification._dims_repr(dims, limit=20)
    assert out.endswith("...")
    assert len(out) == 23  # 20 + "..."


# ---------------------------------------------------------------------------
# aggregate_names
# ---------------------------------------------------------------------------


def _labels(kernels):
    return {"labeled_kernels": kernels}


def test_aggregate_names_duplicates_and_ordering():
    labels = _labels(
        [
            {"name": "small", "dur": 1.0, "perf_category": "Elementwise"},
            {
                "name": "big",
                "dur": 10.0,
                "perf_category": "GEMM",
                "input_dims": [[1, 2]],
            },
            {"name": "big", "dur": 5.0, "perf_category": "GEMM"},
            {"name": "big", "dur": 2.0, "perf_category": "Attention"},
        ]
    )
    agg = kernel_unification.aggregate_names(labels)
    # ordered by descending total duration -> big (17) before small (1)
    assert list(agg) == ["big", "small"]
    big = agg["big"]
    assert big["kernel_count"] == 3
    assert big["total_dur_us"] == 17.0
    # perf_categories deduped + sorted
    assert big["perf_categories"] == ["Attention", "GEMM"]
    # first non-empty input_dims sampled
    assert big["sample_input_dims"] == "[[1,2]]"
    # without key_fn there are no raw-name samples
    assert "sample_raw_names" not in big


def test_aggregate_names_missing_and_none_dur():
    labels = _labels(
        [
            {"dur": None},  # missing name -> "", None dur -> 0
            {"name": "", "dur": 4.0},
        ]
    )
    agg = kernel_unification.aggregate_names(labels)
    assert list(agg) == [""]
    assert agg[""]["kernel_count"] == 2
    assert agg[""]["total_dur_us"] == 4.0
    assert agg[""]["sample_input_dims"] == ""


def test_aggregate_names_key_fn_collects_raw_names():
    labels = _labels(
        [
            {"name": "moe_attn_vllm", "dur": 3.0},
            {"name": "sglang_moe_attention", "dur": 2.0},
            {"name": "moe_attn", "dur": 1.0},  # equals key -> not sampled
        ]
    )
    agg = kernel_unification.aggregate_names(labels, key_fn=lambda n: "moe_attn")
    assert list(agg) == ["moe_attn"]
    entry = agg["moe_attn"]
    assert entry["kernel_count"] == 3
    # raw names differing from the key are recorded and sorted
    assert entry["sample_raw_names"] == ["moe_attn_vllm", "sglang_moe_attention"]


def test_aggregate_names_key_fn_none_drops_kernel():
    labels = _labels(
        [
            {"name": "keep", "dur": 1.0},
            {"name": "drop", "dur": 9.0},
        ]
    )
    agg = kernel_unification.aggregate_names(
        labels, key_fn=lambda n: None if n == "drop" else n
    )
    assert list(agg) == ["keep"]


# ---------------------------------------------------------------------------
# _entry_list / _build_context
# ---------------------------------------------------------------------------


def test_entry_list_preserves_agg_order():
    labels = _labels(
        [
            {"name": "b", "dur": 5.0},
            {"name": "a", "dur": 9.0},
        ]
    )
    agg = kernel_unification.aggregate_names(labels)  # order: a, b
    picked = kernel_unification._entry_list(agg, {"a", "b"})
    assert [e["name"] for e in picked] == ["a", "b"]


def test_build_context_shape_and_extra():
    agg_a = kernel_unification.aggregate_names(
        _labels([{"name": "shared", "dur": 2.0}, {"name": "onlyA", "dur": 1.0}])
    )
    agg_b = kernel_unification.aggregate_names(
        _labels([{"name": "shared", "dur": 2.0}, {"name": "onlyB", "dur": 1.0}])
    )
    ctx = kernel_unification._build_context(
        agg_a, agg_b, "MI300", "B300", "raw_name", extra={"flag": True}
    )
    assert ctx["name_a"] == "MI300"
    assert ctx["key_level"] == "raw_name"
    assert ctx["summary"]["combined_unique"] == 3
    assert ctx["summary"]["in_both"] == 1
    assert ctx["in_both"] == ["shared"]
    assert [e["name"] for e in ctx["only_in_MI300"]] == ["onlyA"]
    assert [e["name"] for e in ctx["only_in_B300"]] == ["onlyB"]
    assert ctx["flag"] is True


def test_build_context_no_extra():
    agg = kernel_unification.aggregate_names(_labels([{"name": "x", "dur": 1.0}]))
    ctx = kernel_unification._build_context(agg, agg, "a", "b", "stem")
    assert ctx["summary"]["combined_unique"] == 1
    assert "flag" not in ctx


# ---------------------------------------------------------------------------
# _sample_names
# ---------------------------------------------------------------------------


def test_sample_names_small_returns_all_tagged():
    agg_a = kernel_unification.aggregate_names(_labels([{"name": "a", "dur": 1.0}]))
    agg_b = kernel_unification.aggregate_names(_labels([{"name": "b", "dur": 1.0}]))
    sample = kernel_unification._sample_names(agg_a, agg_b, "A", "B", 10)
    traces = {row["trace"] for row in sample}
    assert traces == {"A", "B"}
    assert len(sample) == 2


def test_sample_names_spaced_subset():
    agg_a = kernel_unification.aggregate_names(
        _labels([{"name": n, "dur": float(i)} for i, n in enumerate("abcd")])
    )
    agg_b = kernel_unification.aggregate_names(
        _labels([{"name": n, "dur": float(i)} for i, n in enumerate("wxyz")])
    )
    sample = kernel_unification._sample_names(agg_a, agg_b, "A", "B", 2)
    # sample_size 2 -> half=1 from A, 1 from B
    assert len(sample) == 2
    assert [r["trace"] for r in sample] == ["A", "B"]


# ---------------------------------------------------------------------------
# _compile_rules
# ---------------------------------------------------------------------------


def test_compile_rules_valid():
    compiled = kernel_unification._compile_rules(
        [
            {"pattern": r"foo_\d+", "action": "collapse", "replacement": "foo"},
            {"pattern": "bar", "action": "preserve"},
            {"pattern": "baz"},  # default action collapse
        ]
    )
    assert [c["action"] for c in compiled] == ["collapse", "preserve", "collapse"]
    assert compiled[0]["regex"].search("foo_123")


def test_compile_rules_bad_action():
    try:
        kernel_unification._compile_rules([{"pattern": "x", "action": "nope"}])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "invalid action" in str(e)


def test_compile_rules_bad_regex():
    try:
        kernel_unification._compile_rules([{"pattern": "(", "action": "preserve"}])
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "bad regex" in str(e)


# ---------------------------------------------------------------------------
# stem_for
# ---------------------------------------------------------------------------


def test_stem_for_collapse():
    compiled = kernel_unification._compile_rules(
        [{"pattern": r"gemm_[a-z0-9]+", "action": "collapse", "replacement": "gemm"}]
    )
    assert kernel_unification.stem_for("gemm_fp16x8", compiled) == ("gemm", "collapse")


def test_stem_for_preserve_and_drop():
    compiled = kernel_unification._compile_rules(
        [
            {"pattern": "keep", "action": "preserve"},
            {"pattern": "trash", "action": "drop"},
        ]
    )
    assert kernel_unification.stem_for("keep_me", compiled) == ("keep_me", "preserve")
    assert kernel_unification.stem_for("trash_me", compiled) == ("trash_me", "drop")


def test_stem_for_no_match_defaults_preserve():
    compiled = kernel_unification._compile_rules(
        [{"pattern": "zzz", "action": "collapse", "replacement": "z"}]
    )
    assert kernel_unification.stem_for("other", compiled) == ("other", "preserve")


def test_stem_for_bad_replacement_falls_back(capsys):
    # replacement references a non-existent capture group -> re.error at sub time
    compiled = kernel_unification._compile_rules(
        [{"pattern": r"(a)b", "action": "collapse", "replacement": r"\2"}]
    )
    # first call warns and preserves
    assert kernel_unification.stem_for("ab", compiled) == ("ab", "preserve")
    err = capsys.readouterr().err
    assert "ignoring stem rule" in err
    assert compiled[0]["_warned"] is True
    # second call is already warned -> no new warning, still preserves
    assert kernel_unification.stem_for("ab", compiled) == ("ab", "preserve")
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# _load_map_side
# ---------------------------------------------------------------------------


def test_load_map_side_by_side_key():
    doc = {"map_a": {"foo": "bar"}}
    assert kernel_unification._load_map_side(doc, "a", "MI300") == {"foo": "bar"}


def test_load_map_side_by_name_key():
    doc = {"map_MI300": {"foo": "bar"}}
    assert kernel_unification._load_map_side(doc, "a", "MI300") == {"foo": "bar"}


def test_load_map_side_nested_map():
    doc = {"a": {"map": {"foo": "bar"}, "notes": "x"}}
    assert kernel_unification._load_map_side(doc, "a", "MI300") == {"foo": "bar"}


def test_load_map_side_missing_returns_empty():
    assert kernel_unification._load_map_side({}, "a", "MI300") == {}


# ---------------------------------------------------------------------------
# _apply_side
# ---------------------------------------------------------------------------


def test_apply_side_no_stem_map():
    labels = _labels(
        [
            {"name": "foo", "dur": 1.0},
            {"name": "baz", "dur": 1.0},
        ]
    )
    stats = kernel_unification._apply_side(labels, {"foo": "FOO"}, None)
    blocks = [k["semantic_block"] for k in labels["labeled_kernels"]]
    assert blocks == ["FOO", "baz"]  # unmapped falls back to raw
    assert stats == {"kernels": 2, "mapped": 1, "stemmed": 0}


def test_apply_side_with_stem_map():
    labels = _labels(
        [
            {"name": "gemm_v1", "dur": 1.0},
            {"name": "plain", "dur": 1.0},
        ]
    )
    raw_to_stem = {"gemm_v1": "gemm", "plain": "plain"}
    unified = {"gemm": "GEMM_UNIFIED"}
    stats = kernel_unification._apply_side(labels, unified, raw_to_stem)
    blocks = [k["semantic_block"] for k in labels["labeled_kernels"]]
    assert blocks == ["GEMM_UNIFIED", "plain"]
    # gemm_v1 was stemmed (base != raw); plain was not
    assert stats == {"kernels": 2, "mapped": 1, "stemmed": 1}
