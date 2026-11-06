###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the semantic-analysis artifact validators in validation_utils."""

import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from TraceLens.Agent.Analysis.utils.validation_utils import (  # noqa: E402
    _missing_backrefs,
    validate_kernel_coherence_decisions,
    validate_kernel_unification_map,
    validate_stem_rules,
)


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


# ---------------------------------------------------------------------------
# validate_stem_rules
# ---------------------------------------------------------------------------
_GOOD_STEM = {
    "rules": [
        {
            "pattern": r"_MT\d+x\d+x\d+",
            "replacement": "_MT#",
            "action": "collapse",
            "note": "collapse tile-size id",
        },
        {"pattern": "nvtx", "replacement": "", "action": "drop", "note": "noise"},
        {"pattern": "^gemm", "replacement": "", "action": "preserve", "note": "keep"},
    ]
}


def test_stem_rules_valid(tmp_path):
    assert validate_stem_rules(_write(tmp_path, "s.json", _GOOD_STEM)) == (True, [])


def test_stem_rules_bare_list_accepted(tmp_path):
    assert validate_stem_rules(_write(tmp_path, "s.json", _GOOD_STEM["rules"]))[0]


def test_stem_rules_missing_file(tmp_path):
    ok, errs = validate_stem_rules(str(tmp_path / "nope.json"))
    assert ok is False and "File not found" in errs[0]


def test_stem_rules_bad_action(tmp_path):
    doc = {"rules": [{"pattern": "x", "replacement": "", "action": "zap", "note": "n"}]}
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("invalid action" in e for e in errs)


def test_stem_rules_uncompilable_pattern(tmp_path):
    doc = {
        "rules": [
            {"pattern": "gemm_(", "replacement": "g", "action": "collapse", "note": "n"}
        ]
    }
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("bad regex" in e for e in errs)


def test_stem_rules_dangling_backref(tmp_path):
    # replacement references group 2 but pattern defines none
    doc = {
        "rules": [
            {
                "pattern": "ok",
                "replacement": chr(92) + "2",
                "action": "collapse",
                "note": "n",
            }
        ]
    }
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("missing group" in e for e in errs)


def test_stem_rules_empty_note(tmp_path):
    doc = {
        "rules": [{"pattern": "x", "replacement": "", "action": "preserve", "note": ""}]
    }
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("note" in e for e in errs)


def test_stem_rules_duplicate_pattern(tmp_path):
    doc = {
        "rules": [
            {"pattern": "dup", "replacement": "", "action": "preserve", "note": "a"},
            {"pattern": "dup", "replacement": "", "action": "preserve", "note": "b"},
        ]
    }
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("duplicate pattern" in e for e in errs)


# ---------------------------------------------------------------------------
# validate_kernel_unification_map
# ---------------------------------------------------------------------------
def _uni_ctx(tmp_path):
    ctx = {
        "name_a": "A",
        "name_b": "B",
        "only_in_A": [
            {"id": "a0", "name": "gemmA", "perf_categories": ["GEMM"]},
            {"id": "a1", "name": "attnA", "perf_categories": ["Attention"]},
        ],
        "only_in_B": [
            {"id": "b0", "name": "gemmB", "perf_categories": ["GEMM"]},
            {"id": "b1", "name": "attnB", "perf_categories": ["Attention"]},
        ],
        "in_both": [],
    }
    return _write(tmp_path, "ctx.json", ctx)


def test_uni_map_valid(tmp_path):
    ctx = _uni_ctx(tmp_path)
    dec = {
        "pairs": [
            {"a": "a0", "b": "b0", "label": "gemm"},
            {"a": "a1", "b": "b1", "label": "attn"},
        ]
    }
    assert validate_kernel_unification_map(_write(tmp_path, "d.json", dec), ctx) == (
        True,
        [],
    )


def test_uni_map_unknown_id(tmp_path):
    ctx = _uni_ctx(tmp_path)
    dec = {"pairs": [{"a": "a9", "b": "b0", "label": "x"}]}
    ok, errs = validate_kernel_unification_map(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("'a' id 'a9' not in only_in_A" in e for e in errs)


def test_uni_map_cross_perf_category(tmp_path):
    ctx = _uni_ctx(tmp_path)
    dec = {"pairs": [{"a": "a0", "b": "b1", "label": "x"}]}  # GEMM vs Attention
    ok, errs = validate_kernel_unification_map(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("perf_category mismatch" in e for e in errs)


def test_uni_map_reused_id(tmp_path):
    ctx = _uni_ctx(tmp_path)
    dec = {
        "pairs": [
            {"a": "a0", "b": "b0", "label": "x"},
            {"a": "a0", "b": "b1", "label": "y"},
        ]
    }
    ok, errs = validate_kernel_unification_map(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("reused" in e for e in errs)


def test_uni_map_empty_label(tmp_path):
    ctx = _uni_ctx(tmp_path)
    dec = {"pairs": [{"a": "a0", "b": "b0", "label": "  "}]}
    ok, errs = validate_kernel_unification_map(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("non-empty string" in e for e in errs)


def test_uni_map_rename_valid_and_unknown(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, _ = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"renames_a": [{"id": "a0", "label": "x"}]}), ctx
    )
    assert ok is True
    ok2, errs = validate_kernel_unification_map(
        _write(tmp_path, "d2.json", {"renames_b": [{"id": "bX", "label": "x"}]}), ctx
    )
    assert ok2 is False
    assert any("renames_b[0]" in e for e in errs)


def test_uni_map_context_without_ids(tmp_path):
    ctx = _write(
        tmp_path,
        "ctx.json",
        {"name_a": "A", "name_b": "B", "only_in_A": [], "only_in_B": []},
    )
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"pairs": []}), ctx
    )
    assert ok is False
    assert any("no id-stamped candidates" in e for e in errs)


# ---------------------------------------------------------------------------
# validate_kernel_coherence_decisions
# ---------------------------------------------------------------------------
def _coh_ctx(tmp_path):
    ctx = {
        "name_a": "A",
        "name_b": "B",
        "one_sided_in_a": ["symA"],
        "one_sided_in_b": ["symB"],
        "context_catalog": [{"id": "A:0"}, {"id": "B:0"}],
    }
    return _write(tmp_path, "cctx.json", ctx)


def test_coh_valid(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {
        "context_renames": {"A:0": "qkv"},
        "fallback_remap_a": {"symA": "down"},
        "fallback_remap_regex_b": {"^sym": "down"},
        "notes": "ok",
    }
    assert validate_kernel_coherence_decisions(
        _write(tmp_path, "d.json", dec), ctx
    ) == (True, [])


def test_coh_unknown_context_id(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {"context_renames": {"Z:9": "qkv"}}
    ok, errs = validate_kernel_coherence_decisions(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("not in context_catalog" in e for e in errs)


def test_coh_unknown_symbol(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {"fallback_remap_a": {"nope": "down"}}
    ok, errs = validate_kernel_coherence_decisions(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("not in one_sided_in_a" in e for e in errs)


def test_coh_bad_regex(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {"fallback_remap_regex_a": {"(": "down"}}
    ok, errs = validate_kernel_coherence_decisions(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("bad regex" in e for e in errs)


def test_coh_empty_value(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {"fallback_remap_a": {"symA": ""}}
    ok, errs = validate_kernel_coherence_decisions(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("non-empty string" in e for e in errs)


def test_coh_notes_wrong_type(tmp_path):
    ctx = _coh_ctx(tmp_path)
    dec = {"notes": ["not", "a", "string"]}
    ok, errs = validate_kernel_coherence_decisions(_write(tmp_path, "d.json", dec), ctx)
    assert ok is False
    assert any("'notes' must be a string" in e for e in errs)


# ---------------------------------------------------------------------------
# schema / type-guard branches
# ---------------------------------------------------------------------------
def test_corrupt_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json")
    ok, errs = validate_stem_rules(str(p))
    assert ok is False
    assert any("Could not parse" in e for e in errs)


def test_stem_rules_top_level_not_list(tmp_path):
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", {"nope": 1}))
    assert ok is False
    assert any("must be a list" in e for e in errs)


def test_stem_rules_non_dict_rule(tmp_path):
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", {"rules": ["x"]}))
    assert ok is False
    assert any("must be an object" in e for e in errs)


def test_uni_map_pairs_not_list(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"pairs": {"a": "a0"}}), ctx
    )
    assert ok is False
    assert any("'pairs' must be a list" in e for e in errs)


def test_uni_map_non_dict_pair(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"pairs": ["x"]}), ctx
    )
    assert ok is False
    assert any("pair 0: must be an object" in e for e in errs)


def test_uni_map_renames_not_list(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"renames_a": {"id": "a0"}}), ctx
    )
    assert ok is False
    assert any("'renames_a' must be a list" in e for e in errs)


def test_uni_map_non_dict_rename(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"renames_a": ["x"]}), ctx
    )
    assert ok is False
    assert any("renames_a[0]: must be an object" in e for e in errs)


def test_coh_map_not_object(tmp_path):
    ctx = _coh_ctx(tmp_path)
    ok, errs = validate_kernel_coherence_decisions(
        _write(tmp_path, "d.json", {"context_renames": ["x"]}), ctx
    )
    assert ok is False
    assert any("'context_renames' must be an object" in e for e in errs)


def test_coh_missing_decisions_file(tmp_path):
    ctx = _coh_ctx(tmp_path)
    ok, errs = validate_kernel_coherence_decisions(str(tmp_path / "nope.json"), ctx)
    assert ok is False
    assert any("File not found" in e for e in errs)


def test_uni_map_missing_context_file(tmp_path):
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"pairs": []}), str(tmp_path / "nope.json")
    )
    assert ok is False
    assert any("File not found" in e for e in errs)


def test_uni_map_missing_decisions_file(tmp_path):
    ok, errs = validate_kernel_unification_map(
        str(tmp_path / "nope.json"), _uni_ctx(tmp_path)
    )
    assert ok is False
    assert any("File not found" in e for e in errs)


def test_uni_map_bad_and_reused_b_id(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"pairs": [{"a": "a0", "b": "b9", "label": "x"}]}),
        ctx,
    )
    assert ok is False
    assert any("'b' id 'b9' not in only_in_B" in e for e in errs)
    ok2, errs2 = validate_kernel_unification_map(
        _write(
            tmp_path,
            "d2.json",
            {
                "pairs": [
                    {"a": "a0", "b": "b0", "label": "x"},
                    {"a": "a1", "b": "b0", "label": "y"},
                ]
            },
        ),
        ctx,
    )
    assert ok2 is False
    assert any("'b' id 'b0' reused" in e for e in errs2)


def test_uni_map_rename_empty_label(tmp_path):
    ctx = _uni_ctx(tmp_path)
    ok, errs = validate_kernel_unification_map(
        _write(tmp_path, "d.json", {"renames_a": [{"id": "a0", "label": ""}]}), ctx
    )
    assert ok is False
    assert any("renames_a[0]: 'label'" in e for e in errs)


def test_coh_missing_context_file(tmp_path):
    ok, errs = validate_kernel_coherence_decisions(
        _write(tmp_path, "d.json", {}), str(tmp_path / "nope.json")
    )
    assert ok is False
    assert any("File not found" in e for e in errs)


def test_stem_rules_missing_pattern(tmp_path):
    doc = {"rules": [{"replacement": "", "action": "preserve", "note": "n"}]}
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("'pattern' must be a non-empty string" in e for e in errs)


def test_stem_rules_non_string_replacement(tmp_path):
    doc = {
        "rules": [{"pattern": "x", "replacement": 5, "action": "collapse", "note": "n"}]
    }
    ok, errs = validate_stem_rules(_write(tmp_path, "s.json", doc))
    assert ok is False
    assert any("'replacement' must be a string" in e for e in errs)


# ---------------------------------------------------------------------------
# _missing_backrefs (named groups)
# ---------------------------------------------------------------------------
def test_missing_backrefs_named_groups():
    import re

    bs = chr(92)
    # named ref to a group that exists -> fine
    assert _missing_backrefs(bs + "g<x>", re.compile("(?P<x>ok)")) == set()
    # named ref to a missing group -> flagged
    assert _missing_backrefs(bs + "g<nope>", re.compile("(ok)")) == {"nope"}
    # numeric-in-\g<> beyond group count -> flagged
    assert _missing_backrefs(bs + "g<3>", re.compile("(ok)")) == {"3"}
