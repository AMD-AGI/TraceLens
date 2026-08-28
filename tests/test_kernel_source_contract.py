###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tests for the source-resolution audit artifact (contract)."""

from TraceLens.TraceUtils.kernel_source import contract


def test_make_entry_has_all_required_keys():
    entry = contract.make_entry(kernel_id="k001", name="add_kernel", gpu_pct=12.5)
    for key in contract.REQUIRED_ENTRY_KEYS:
        assert key in entry
    assert entry["method"] == contract.METHOD_UNRESOLVED


def test_valid_document_has_no_problems():
    entry = contract.make_entry(
        kernel_id="k001",
        name="add_kernel",
        gpu_pct=12.5,
        source_file="/pkg/csrc/ops.cu",
        source_line=42,
        method=contract.METHOD_SYMBOL_INDEX,
    )
    doc = contract.make_document([entry], generated_by="unit-test")
    assert contract.validate_document(doc) == []


def test_gate_non_patchable_entry_may_omit_source():
    entry = contract.make_entry(
        kernel_id="k002",
        name="Cijk_Ailk_Bljk",
        gpu_pct=3.0,
        method=contract.METHOD_GATE_NON_PATCHABLE,
        reason="tensile_precompiled",
    )
    doc = contract.make_document([entry], generated_by="unit-test")
    assert contract.validate_document(doc) == []


def test_source_without_method_flagged():
    # A resolved-looking method with no source_file is a contract violation.
    entry = contract.make_entry(
        kernel_id="k003",
        name="foo",
        gpu_pct=1.0,
        method=contract.METHOD_SYMBOL_INDEX,
    )
    problems = contract.validate_document(
        contract.make_document([entry], generated_by="t")
    )
    assert any("no source_file" in p for p in problems)


def test_unknown_method_flagged():
    entry = contract.make_entry(
        kernel_id="k004", name="foo", gpu_pct=1.0, method="made_up"
    )
    entry["source_file"] = "/x.cu"
    problems = contract.validate_document(
        contract.make_document([entry], generated_by="t")
    )
    assert any("unknown method" in p for p in problems)


def test_invalid_confidence_flagged():
    entry = contract.make_entry(
        kernel_id="k005",
        name="foo",
        gpu_pct=1.0,
        source_file="/x.cu",
        method=contract.METHOD_LLM,
        confidence=1.7,
    )
    problems = contract.validate_document(
        contract.make_document([entry], generated_by="t")
    )
    assert any("confidence" in p for p in problems)


def test_round_trip_read_document(tmp_path):
    import json

    entry = contract.make_entry(kernel_id="k006", name="foo", gpu_pct=1.0)
    doc = contract.make_document([entry], generated_by="unit-test")
    path = tmp_path / contract.SOURCE_RESOLUTION_FILENAME
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded = contract.read_document(path)
    assert loaded is not None
    assert contract.validate_document(loaded) == []


def test_split_line_suffix():
    path, line, func = contract.split_line_suffix("/repo/moe.py(247): _grouped_gemm")
    assert path == "/repo/moe.py"
    assert line == 247
    assert func == "_grouped_gemm"
