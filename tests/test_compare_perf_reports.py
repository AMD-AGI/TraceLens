###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Tests for the compare_perf_reports_pytorch tool.

For each test case, compares two performance reports and validates the generated
comparison against a reference output. Sheets to compare are driven by whatever
sheets exist in the reference xlsx.
"""

import os
import shutil

import pandas as pd
import pytest

from TraceLens.Reporting.generate_perf_report_pytorch import (
    generate_perf_report_pytorch,
)
from TraceLens.Reporting.compare_perf_reports_pytorch import (
    generate_compare_perf_reports_pytorch,
)

from conftest import compare_cols, format_diff_details


def find_compare_test_cases(root="tests/traces"):
    """
    Find compare test case directories. Each must contain:
      - Two input reports
      - A reference/ directory with the expected comparison xlsx
    Returns list of (test_dir, report1, report2, ref_xlsx, names, input_sheets) tuples.
    input_sheets are passed to generate_compare_perf_reports_pytorch; the output
    sheet names (read from the reference xlsx) are used for comparison.
    """
    test_cases = []

    # compare_test: 56ch_rank7 with kernel_summary
    test_dir = os.path.join(root, "compare_test")
    r1 = os.path.join(test_dir, "256thread", "perf_56ch_rank7.xlsx")
    r2 = os.path.join(test_dir, "512thread", "perf_56ch_rank7.xlsx")
    ref = os.path.join(test_dir, "reference", "compare_56ch_rank7_across_threads.xlsx")
    if all(os.path.exists(p) for p in [r1, r2, ref]):
        test_cases.append(
            (
                test_dir,
                r1,
                r2,
                ref,
                ["256thread", "512thread"],
                ["gpu_timeline", "kernel_summary"],
            )
        )

    # compare_test_ops: 28ch_rank0 with ops_summary
    test_dir = os.path.join(root, "compare_test_ops")
    r1 = os.path.join(test_dir, "256thread", "perf_28ch_rank0.xlsx")
    r2 = os.path.join(test_dir, "512thread", "perf_28ch_rank0.xlsx")
    ref = os.path.join(test_dir, "reference", "compare_28ch_rank0_across_threads.xlsx")
    if all(os.path.exists(p) for p in [r1, r2, ref]):
        test_cases.append(
            (
                test_dir,
                r1,
                r2,
                ref,
                ["256thread", "512thread"],
                ["gpu_timeline", "ops_summary"],
            )
        )

    return test_cases


@pytest.mark.parametrize(
    "test_dir,report1,report2,ref_xlsx,names,input_sheets",
    find_compare_test_cases(),
)
def test_compare_perf_reports(
    test_dir, report1, report2, ref_xlsx, names, input_sheets, tol=1e-6
):
    """
    Generate a comparison report and validate against reference.
    input_sheets are passed to the compare tool; output sheets from the
    reference xlsx are used for column-by-column comparison.
    """
    ref_output_sheets = pd.ExcelFile(ref_xlsx).sheet_names

    fn_root = os.path.join(test_dir, "pytest_output")
    os.makedirs(fn_root, exist_ok=True)

    try:
        fn_comparison_path = os.path.join(fn_root, "comparison_output.xlsx")
        generate_compare_perf_reports_pytorch(
            reports=[report1, report2],
            output=fn_comparison_path,
            names=names,
            sheets=input_sheets,
        )

        assert os.path.exists(
            fn_comparison_path
        ), f"Comparison output not created: {fn_comparison_path}"

        for sheet in ref_output_sheets:
            try:
                df_ref = pd.read_excel(ref_xlsx, sheet_name=sheet)
                df_fn = pd.read_excel(fn_comparison_path, sheet_name=sheet)
            except ValueError as e:
                pytest.fail(f"Sheet '{sheet}' not found: {e}")

            if df_ref.empty:
                assert (
                    df_fn.empty
                ), f"Reference sheet '{sheet}' is empty but generated has {len(df_fn)} rows"
                continue

            assert set(df_ref.columns) == set(
                df_fn.columns
            ), f"Sheet '{sheet}' has different columns"

            cols = df_ref.columns.tolist()
            diff_cols = compare_cols(df_fn, df_ref, cols, tol=tol)
            assert (
                not diff_cols
            ), f"Sheet '{sheet}' has differences: {format_diff_details(diff_cols)}"

    finally:
        if os.path.exists(fn_root):
            shutil.rmtree(fn_root)


# ─── E2E compare report tests (generate reports from traces, then compare) ────

E2E_TRACE_NAME = "Qwen_Qwen1.5-0.5B-Chat__1016005"
E2E_TEST_DIR = os.path.join("tests", "traces", "compare_test_e2e")
E2E_REF_DIR = os.path.join(E2E_TEST_DIR, "reference")

E2E_H100_TRACE = os.path.join("tests", "traces", "h100", f"{E2E_TRACE_NAME}.json.gz")
E2E_MI300_TRACE = os.path.join("tests", "traces", "mi300", f"{E2E_TRACE_NAME}.json.gz")

E2E_REF_H100_REPORT = os.path.join(E2E_REF_DIR, "h100_perf_report.xlsx")
E2E_REF_MI300_REPORT = os.path.join(E2E_REF_DIR, "mi300_perf_report.xlsx")
E2E_REF_COMPARE_REPORT = os.path.join(E2E_REF_DIR, "compare_h100_mi300_qwen.xlsx")

E2E_COLS_TO_SKIP = {"Input Dims", "Input Strides", "Concrete Inputs", "Input type"}


def _validate_report_against_reference(generated_path, reference_path, tol=1e-6):
    """Validate a generated report against its reference, sheet by sheet."""
    ref_sheets = pd.ExcelFile(reference_path).sheet_names
    gen_sheets = pd.ExcelFile(generated_path).sheet_names

    missing = set(ref_sheets) - set(gen_sheets)
    assert not missing, f"Missing sheets in generated report: {missing}"

    errors = []
    for sheet in ref_sheets:
        df_ref = pd.read_excel(reference_path, sheet_name=sheet)
        df_gen = pd.read_excel(generated_path, sheet_name=sheet)

        if df_ref.empty:
            if not df_gen.empty:
                errors.append(
                    f"Sheet '{sheet}': reference is empty but generated has "
                    f"{len(df_gen)} rows"
                )
            continue

        missing_cols = set(df_ref.columns) - set(df_gen.columns)
        if missing_cols:
            errors.append(
                f"Sheet '{sheet}': columns missing from generated report: "
                f"{missing_cols}"
            )
            continue

        cols = [c for c in df_ref.columns if c not in E2E_COLS_TO_SKIP]
        diff_cols = compare_cols(df_gen, df_ref, cols, tol=tol)
        if diff_cols:
            errors.append(
                f"Sheet '{sheet}': value differences:"
                f"{format_diff_details(diff_cols)}"
            )

    return errors


@pytest.fixture(scope="module")
def generated_reports(tmp_path_factory):
    """Generate perf reports from both traces once for all tests in this module."""
    output_dir = tmp_path_factory.mktemp("compare_e2e")

    h100_report = str(output_dir / "h100_perf_report.xlsx")
    mi300_report = str(output_dir / "mi300_perf_report.xlsx")

    generate_perf_report_pytorch(
        profile_json_path=E2E_H100_TRACE,
        output_xlsx_path=h100_report,
        kernel_summary=True,
        short_kernel_study=True,
    )
    generate_perf_report_pytorch(
        profile_json_path=E2E_MI300_TRACE,
        output_xlsx_path=mi300_report,
        kernel_summary=True,
        short_kernel_study=True,
    )

    return {"h100": h100_report, "mi300": mi300_report, "output_dir": output_dir}


def test_individual_reports_match_reference(generated_reports):
    """Verify that generated perf reports match reference outputs."""
    for tag, ref_path in [
        ("h100", E2E_REF_H100_REPORT),
        ("mi300", E2E_REF_MI300_REPORT),
    ]:
        gen_path = generated_reports[tag]
        errors = _validate_report_against_reference(gen_path, ref_path)
        assert not errors, f"{tag} report differences:\n" + "\n".join(errors)


def test_compare_report_sheets_present(generated_reports):
    """Verify the comparison report contains all expected sheets."""
    output_dir = generated_reports["output_dir"]
    compare_path = str(output_dir / "compare_reports.xlsx")

    generate_compare_perf_reports_pytorch(
        reports=[generated_reports["h100"], generated_reports["mi300"]],
        output=compare_path,
        names=["h100", "mi300"],
        sheets=["all"],
    )

    ref_sheets = set(pd.ExcelFile(E2E_REF_COMPARE_REPORT).sheet_names)
    gen_sheets = set(pd.ExcelFile(compare_path).sheet_names)

    missing = ref_sheets - gen_sheets
    assert not missing, f"Missing sheets in compare report: {missing}"


def test_compare_report_matches_reference(generated_reports):
    """Validate comparison report contents against reference."""
    output_dir = generated_reports["output_dir"]
    compare_path = str(output_dir / "compare_reports.xlsx")

    if not os.path.exists(compare_path):
        generate_compare_perf_reports_pytorch(
            reports=[generated_reports["h100"], generated_reports["mi300"]],
            output=compare_path,
            names=["h100", "mi300"],
            sheets=["all"],
        )

    errors = _validate_report_against_reference(compare_path, E2E_REF_COMPARE_REPORT)
    assert not errors, "Compare report differences:\n" + "\n".join(errors)
