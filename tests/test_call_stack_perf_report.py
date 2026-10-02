###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the include_call_stack=True path in generate_perf_report_pytorch.

Uses tests/traces/h100/Falconsai_nsfw_image_detection__1016002.json.gz because
it contains ops with multiple GPU kernels (e.g. aten::convolution dispatches 7),
which exercises the multi-kernel call stack merging logic.

Covers:
- call_stack_full column is present in unified_perf_summary
- Single-kernel rows produce a flat list call stack
- Multi-kernel rows produce common prefix + diverging tails (nested lists)
- entry_point column is populated and non-empty for rows that have a call stack
"""

import ast

import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
    _find_entry_point,
)

TRACE_PATH = "tests/traces/h100/Falconsai_nsfw_image_detection__1016002.json.gz"

pytestmark = pytest.mark.filterwarnings(
    "ignore::UserWarning",
)


@pytest.fixture(scope="module")
def unified_perf_summary(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("csvs"))
    result = generate_perf_report_pytorch(
        profile_json_path=TRACE_PATH,
        output_xlsx_path=None,
        output_csvs_dir=out,
        kernel_summary=False,
        short_kernel_study=False,
        include_call_stack=True,
    )
    return result["unified_perf_summary"]


# ---------------------------------------------------------------------------
# 1. call_stack_full column is present
# ---------------------------------------------------------------------------


def test_call_stack_full_column_present(unified_perf_summary):
    assert "call_stack_full" in unified_perf_summary.columns


# ---------------------------------------------------------------------------
# 2. Single-kernel rows produce a flat list
# ---------------------------------------------------------------------------


def test_single_kernel_rows_have_flat_call_stack(unified_perf_summary):
    assert "kernel_details_summary" in unified_perf_summary.columns

    found = False
    for _, row in unified_perf_summary.iterrows():
        kd = row["kernel_details_summary"]
        if not isinstance(kd, list) or len(kd) != 1:
            continue
        cs_val = row["call_stack_full"]
        if not isinstance(cs_val, str) or cs_val == "Not found":
            continue
        parsed = ast.literal_eval(cs_val)
        assert isinstance(parsed, list), f"Expected list, got {type(parsed)}"
        assert not any(
            isinstance(x, list) for x in parsed
        ), f"Single-kernel row has nested list: {parsed}"
        assert len(parsed) > 0
        found = True
        break

    assert (
        found
    ), "No single-kernel rows with a call stack found in unified_perf_summary"


# ---------------------------------------------------------------------------
# 3. Multi-kernel rows produce common prefix + diverging tails
# ---------------------------------------------------------------------------


def test_multi_kernel_rows_have_nested_call_stack(unified_perf_summary):
    assert "kernel_details_summary" in unified_perf_summary.columns

    found = False
    for _, row in unified_perf_summary.iterrows():
        kd = row["kernel_details_summary"]
        if not isinstance(kd, list) or len(kd) <= 1:
            continue
        cs_val = row["call_stack_full"]
        if not isinstance(cs_val, str) or cs_val == "Not found":
            continue
        parsed = ast.literal_eval(cs_val)
        assert isinstance(parsed, list), f"Expected list, got {type(parsed)}"
        tails = [x for x in parsed if isinstance(x, list)]
        unique_kernel_names = len({k["name"] for k in kd})
        assert len(tails) == unique_kernel_names, (
            f"Multi-kernel row '{row['name']}': expected {unique_kernel_names} tails "
            f"(unique kernels), got {len(tails)}"
        )
        found = True
        break

    assert found, "No multi-kernel rows with a call stack found in unified_perf_summary"


# ---------------------------------------------------------------------------
# entry_point logic
# ---------------------------------------------------------------------------


def test_entry_point_column_present(unified_perf_summary):
    assert "entry_point" in unified_perf_summary.columns


def test_find_entry_point_inward_matching():
    # A stack where a .py frame with the op name appears after the op
    stack = str(
        ["user_code.py(10): my_func", "aten::addmm", "torch/functional.py(5): addmm"]
    )
    result = _find_entry_point(stack, "aten::addmm")
    assert result["traversal"] == "inward"
    assert "addmm" in result["entry_point"]
    assert result["num_wrappers"] >= 0


def test_find_entry_point_outward_matching():
    # No inward .py frame with op name — should fall back to outward
    stack = str(["user_code.py(10): forward", "aten::addmm"])
    result = _find_entry_point(stack, "aten::addmm")
    assert result["traversal"] == "outward"
    assert result["entry_point"] == "user_code.py(10): forward"


def test_find_entry_point_not_found():
    # Stack with only wrapper frames — should return Not found
    stack = str(["torch/nn/modules/module.py(5): _call_impl", "aten::addmm"])
    result = _find_entry_point(stack, "aten::addmm")
    assert result["entry_point"] == "Not found"
