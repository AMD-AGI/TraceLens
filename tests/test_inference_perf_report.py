###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Regression tests for generate_perf_report_pytorch_inference.
# Each test case is a subdirectory under tests/traces/inference/ containing:
#   - A .json.gz trace file
#   - A perf_csvs/ folder with reference CSV files (one per output sheet)
#   - Optionally capture_traces/ (graph capture mode)
#   - Optionally gpu_arch.json

import os, numpy as np, pandas as pd, pytest, ast, re, gzip, json
from pandas.api.types import is_float_dtype
from TraceLens.Reporting.generate_perf_report_pytorch_inference import (
    classify_graph_capture_trace,
    generate_perf_report_pytorch,
    generate_perf_report_pytorch as gen_inf,
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_capture_to_graph,
    _align_graph_to_capture_by_group,
    _capture_tree_cache,
    align_streams,
    capture_has_kernel_names,
    merge_capture_trace_into_graph,
    verify_subtree_events,
)
from conftest import update_reference_csvs
from tests.fixtures.traces import INFERENCE_ROOT, _discover_inference_cases
from tests.fixtures.reporting import _build_synthetic_trace, _mk_event, _write_trace

pytestmark = pytest.mark.filterwarnings(
    "ignore:Input list of events is empty.*:UserWarning",
    "ignore:Input DataFrame is empty.*:UserWarning",
    "ignore:Source column 'kernel_details__summarize_kernel_stats' not found.*:UserWarning",
    "ignore:Found .* events with failed performance metric computation.*:UserWarning",
    "ignore:Inconsistent kernel list length found.*:UserWarning",
    "ignore:There are hipgraph launches.*:UserWarning",
)

INFERENCE_TRACES_ROOT = "tests/traces/inference"


# ---------------------------------------------------------------------------
# Helpers (shared with test_compare_perf_report.py — kept self-contained here
# so this file can run independently)
# ---------------------------------------------------------------------------


def normalize_value(val):
    """Convert numpy scalars and their string representations to Python native types."""
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    elif isinstance(val, list):
        return [normalize_value(v) for v in val]
    elif isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    elif isinstance(val, str):
        cleaned_val = re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", val)
        try:
            parsed = ast.literal_eval(cleaned_val)
            return normalize_value(parsed)
        except (ValueError, SyntaxError):
            return val
    return val


UID_DIFF_TOLERANCE = 10


def compare_cols(df_test, df_ref, cols, tol=1e-6):
    """Compare columns in two dataframes, skipping rows where ref is None/NaN."""
    diff_details = {}
    for col in cols:
        valid_mask = df_ref[col].notna()
        if not valid_mask.any():
            continue
        ref_col = df_ref.loc[valid_mask, col]
        test_col = df_test.loc[df_test.index.intersection(ref_col.index), col]
        test_col, ref_col = test_col.align(ref_col, join="right")

        test_col = test_col.apply(normalize_value)
        ref_col = ref_col.apply(normalize_value)
        print(f"Column: {col}")
        if "UID" in col:
            # Allow UID values to differ by up to UID_DIFF_TOLERANCE (synthetic
            # ops now receive fresh UIDs that may shift relative to the reference).
            try:
                diff = test_col.astype(float) - ref_col.astype(float)
                mismatch = diff.abs() >= UID_DIFF_TOLERANCE
                if mismatch.any():
                    diff_indices = mismatch[mismatch].index.tolist()
                    diff_details[col] = {
                        "num_diffs": len(diff_indices),
                        "sample_diffs": [
                            (idx, test_col[idx], ref_col[idx])
                            for idx in diff_indices[:5]
                        ],
                    }
            except (TypeError, ValueError):
                # Fall back to exact comparison if values aren't numeric
                mismatch = test_col != ref_col
                if mismatch.any():
                    diff_indices = mismatch[mismatch].index.tolist()
                    diff_details[col] = {
                        "num_diffs": len(diff_indices),
                        "sample_diffs": [
                            (idx, test_col[idx], ref_col[idx])
                            for idx in diff_indices[:5]
                        ],
                    }
        elif is_float_dtype(df_test[col]):
            diff = test_col - ref_col
            max_diff = diff.abs().max()
            if not max_diff < tol:
                diff_indices = diff[diff.abs() >= tol].index.tolist()
                diff_details[col] = {
                    "max_diff": max_diff,
                    "num_diffs": len(diff_indices),
                    "sample_diffs": [
                        (idx, test_col[idx], ref_col[idx], diff[idx])
                        for idx in diff_indices[:5]
                    ],
                }
        else:
            mismatch = test_col != ref_col
            if mismatch.any():
                diff_indices = mismatch[mismatch].index.tolist()
                diff_details[col] = {
                    "num_diffs": len(diff_indices),
                    "sample_diffs": [
                        (idx, test_col[idx], ref_col[idx]) for idx in diff_indices[:5]
                    ],
                }
    return diff_details


def format_diff_details(diff_details):
    """Format difference details for readable assertion messages."""
    lines = []
    for col, details in diff_details.items():
        lines.append(f"\n  Column: '{col}'")
        lines.append(f"    Total differences: {details['num_diffs']}")

        if "max_diff" in details:
            lines.append(f"    Max difference: {details['max_diff']:.6e}")
            lines.append("    Sample differences:")
            lines.append(
                f"      {'Index':<8} {'Test Value':<20} {'Ref Value':<20} {'Difference':<15}"
            )
            lines.append(f"      {'-'*8} {'-'*20} {'-'*20} {'-'*15}")
            for idx, test_val, ref_val, diff in details["sample_diffs"]:
                lines.append(
                    f"      {idx:<8} {test_val:<20.6e} {ref_val:<20.6e} {diff:<15.6e}"
                )
        else:
            lines.append("    Sample differences:")
            lines.append(f"      {'Index':<8} {'Test Value':<30} {'Ref Value':<30}")
            lines.append(f"      {'-'*8} {'-'*30} {'-'*30}")
            for idx, test_val, ref_val in details["sample_diffs"]:
                lines.append(f"      {idx:<8} {str(test_val):<30} {str(ref_val):<30}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test case discovery
# ---------------------------------------------------------------------------

COLS_IGNORE = [
    "Non-Data-Mov TFLOPS/s_mean",
    "Non-Data-Mov Kernel Time (µs)_sum",
    "Non-Data-Mov Kernel Time (µs)_mean",
]


def find_inference_test_cases():
    """
    Discover test cases under INFERENCE_TRACES_ROOT.
    Each subdirectory with a .json.gz and a perf_csvs/ folder becomes a test case.
    """
    test_cases = []
    if not os.path.isdir(INFERENCE_TRACES_ROOT):
        return test_cases
    for entry in sorted(os.listdir(INFERENCE_TRACES_ROOT)):
        dirpath = os.path.join(INFERENCE_TRACES_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz_files = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        perf_csvs_dir = os.path.join(dirpath, "perf_csvs")
        if not gz_files or not os.path.isdir(perf_csvs_dir):
            continue
        trace_gz = gz_files[0]
        capture_folder = os.path.join(dirpath, "capture_traces")
        if not os.path.isdir(capture_folder):
            capture_folder = None
        gpu_arch = os.path.join(dirpath, "gpu_arch.json")
        if not os.path.isfile(gpu_arch):
            gpu_arch = None
        test_cases.append(
            pytest.param(dirpath, trace_gz, capture_folder, gpu_arch, id=entry)
        )
    return test_cases


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_capture_tree_cache():
    # Reproduce per-process isolation: the capture-tree cache is keyed by
    # {batch_size}_{mode}, which collides across fixtures sharing a key.
    _capture_tree_cache.clear()
    yield
    _capture_tree_cache.clear()


@pytest.mark.parametrize(
    "dirpath,trace_gz,capture_folder,gpu_arch_path",
    find_inference_test_cases(),
)
def test_inference_perf_report(
    dirpath,
    trace_gz,
    capture_folder,
    gpu_arch_path,
    tmp_path,
    update_references,
    tol=1e-6,
):
    """
    Directly call generate_perf_report_pytorch (from the inference module)
    and compare every output CSV against the reference CSVs in perf_csvs/.
    """
    profile_path = os.path.join(dirpath, trace_gz)
    ref_csvs_dir = os.path.join(dirpath, "perf_csvs")

    # Build the augmented graph tree when capture traces are present
    graph_tree = None
    if capture_folder:
        classify_graph_capture_trace(capture_folder)
        metadata_json_path = os.path.join(capture_folder, "execution_details.json")
        graph_tree = merge_capture_trace_into_graph(
            capture_folder, metadata_json_path, profile_path
        )

    # Call the function under test — write CSVs to a temp directory
    output_csvs_dir = str(tmp_path / "csvs")
    os.makedirs(output_csvs_dir, exist_ok=True)
    result = generate_perf_report_pytorch(
        profile_json_path=profile_path,
        augmented_tree=graph_tree,
        output_xlsx_path=None,
        output_csvs_dir=output_csvs_dir,
        enable_pseudo_ops=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        collective_analysis=False,
        include_call_stack=True,
    )

    # result is a dict[str, pd.DataFrame]
    assert isinstance(result, dict), "generate_perf_report_pytorch must return a dict"
    assert len(result) > 0, "Result dict must not be empty"
    assert "gpu_timeline" in result, "gpu_timeline sheet must always be present"

    # Validate returned DataFrames are well-formed
    for sheet_name, df in result.items():
        assert isinstance(df, pd.DataFrame), f"Sheet '{sheet_name}' is not a DataFrame"
        assert not df.empty, f"Sheet '{sheet_name}' is unexpectedly empty"

    # Verify CSV output was written for every sheet
    for sheet_name in result:
        csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
        assert os.path.exists(
            csv_path
        ), f"CSV output for sheet '{sheet_name}' was not written to {csv_path}"

    if update_references:
        update_reference_csvs(output_csvs_dir, ref_csvs_dir)
        return

    # Compare each generated CSV against the reference CSV in perf_csvs/
    ref_csv_files = [f for f in os.listdir(ref_csvs_dir) if f.endswith(".csv")]
    for csv_file in ref_csv_files:
        sheet_name = csv_file[: -len(".csv")]
        ref_csv_path = os.path.join(ref_csvs_dir, csv_file)
        df_ref = pd.read_csv(ref_csv_path)
        if df_ref.empty:
            continue
        assert sheet_name in result, (
            f"'{sheet_name}' exists in reference perf_csvs/ but was not returned by "
            f"generate_perf_report_pytorch"
        )
        df_test = result[sheet_name]
        cols = [
            col
            for col in df_ref.columns
            if col not in COLS_IGNORE and col in df_test.columns
        ]
        diff_cols = compare_cols(df_test, df_ref, cols, tol=tol)
        assert not diff_cols, (
            f"'{sheet_name}' has differences in {profile_path}:"
            f"{format_diff_details(diff_cols)}"
        )


INFERENCE_ROOT = os.path.join(os.path.dirname(__file__), "traces/inference")


def _discover_cases():
    if not os.path.isdir(INFERENCE_ROOT):
        return []
    cases = []
    for entry in sorted(os.listdir(INFERENCE_ROOT)):
        dirpath = os.path.join(INFERENCE_ROOT, entry)
        if not os.path.isdir(dirpath):
            continue
        gz = [f for f in os.listdir(dirpath) if f.endswith(".json.gz")]
        if not gz:
            continue
        capture = os.path.join(dirpath, "capture_traces")
        cases.append(
            pytest.param(
                dirpath,
                gz[0],
                capture if os.path.isdir(capture) else None,
                id=entry,
            )
        )
    return cases


@pytest.mark.parametrize("dirpath,trace_gz,capture_folder", _discover_cases())
def test_inference_report_extended_flags(dirpath, trace_gz, capture_folder, tmp_path):
    trace_path = os.path.join(dirpath, trace_gz)
    out = tmp_path / "out"
    generate_perf_report_pytorch(
        profile_json_path=trace_path,
        output_csvs_dir=str(out),
        output_xlsx_path=str(tmp_path / "report.xlsx"),
        collective_analysis=True,
        kernel_summary=True,
        short_kernel_study=True,
        include_overlap_info=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        enable_pseudo_ops=True,
        micro_idle_thresh_us=1,
    )
    assert (out / "gpu_timeline.csv").exists()


@pytest.mark.parametrize("dirpath,trace_gz,capture_folder", _discover_cases())
def test_merge_capture_trace_integration(dirpath, trace_gz, capture_folder):
    if capture_folder is None:
        pytest.skip("no capture traces")
    metadata = os.path.join(capture_folder, "execution_details.json")
    if not os.path.isfile(metadata):
        pytest.skip("no execution_details.json")
    trace_path = os.path.join(dirpath, trace_gz)
    merged = merge_capture_trace_into_graph(capture_folder, metadata, trace_path)
    assert len(merged.events) > 0


class TestCaptureMergeHelpers:
    def test_align_capture_to_graph_memcpy(self):
        capture = [{"name": "cudaMemcpy", "args": {}}]
        graph = [{"name": "MemcpyHtoD", "args": {}}]
        aligned = _align_capture_to_graph(capture, graph)
        assert aligned is not None

    def test_align_capture_to_graph_mismatch(self):
        capture = [{"name": "hipLaunchKernel", "args": {"kernel": "a"}}]
        graph = [{"name": "b", "args": {}}]
        assert _align_capture_to_graph(capture, graph) is None

    def test_align_graph_to_capture_group_mismatch(self):
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        ]
        graph = [{"name": "k1", "args": {}}]
        assert _align_graph_to_capture_by_group(capture, graph) is None

    def test_align_streams_and_capture_has_names(self):
        graph = [
            {"name": "k1", "args": {"stream": 1}},
            {"name": "k2", "args": {"stream": 2}},
        ]
        capture = [
            {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
            {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
        ]
        assert capture_has_kernel_names(capture)
        aligned = align_streams(graph, capture)
        assert aligned is not None

    def test_capture_missing_kernel_name(self):
        capture = [{"name": "hipLaunchKernel", "args": {}}]
        assert capture_has_kernel_names(capture) is False


class TestReportingInferenceSheets:
    def test_inference_all_report_variants(self, tmp_path):

        trace = _write_trace(
            tmp_path,
            [
                ("aten::mm", "gemm_kernel", 100),
                ("aten::add", "vectorized_elementwise_kernel", 20),
                ("aten::native_layer_norm", "layer_norm_kernel", 30),
            ],
        )
        gen_inf(
            profile_json_path=trace,
            output_csvs_dir=str(tmp_path / "out"),
            output_xlsx_path=str(tmp_path / "r.xlsx"),
            collective_analysis=True,
            enable_pseudo_ops=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            topk_ops=10,
            topk_roofline_ops=5,
            include_unlinked_kernels=True,
            include_call_stack=True,
        )
        assert (tmp_path / "out" / "gpu_timeline.csv").exists()


class TestInferenceZipPhase8:
    def test_classify_graph_capture_json_gz(self, tmp_path):
        capture_dir = tmp_path / "cap"
        capture_dir.mkdir()
        events = {
            "traceEvents": [
                _mk_event(
                    "cpu_op",
                    "vllm/v1/worker/gpu_model_runner.py(1): _dummy_run",
                    1000,
                    50,
                    1,
                    1,
                    {},
                ),
                _mk_event("cuda_runtime", "cudaStreamBeginCapture", 1100, 10, 1, 1, {}),
                _mk_event(
                    "cpu_op",
                    "aten::mm",
                    1200,
                    20,
                    1,
                    1,
                    {"Input Dims": [[4, 8], [8, 16]]},
                ),
            ]
        }
        gz_path = capture_dir / "graph_capture_rank_0.json.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            json.dump(events, f)
        classify_graph_capture_trace(str(capture_dir))
        details = json.loads((capture_dir / "execution_details.json").read_text())
        assert details[0]["batch_size"] == 4


@pytest.mark.parametrize("dirpath,trace_gz", _discover_inference_cases())
def test_inference_fixture_full_report(dirpath, trace_gz, tmp_path):
    trace_path = os.path.join(dirpath, trace_gz)
    out = tmp_path / "csv"
    result = generate_inference_report(
        profile_json_path=trace_path,
        output_csvs_dir=str(out),
        output_xlsx_path=str(tmp_path / "report.xlsx"),
        collective_analysis=False,
        enable_pseudo_ops=True,
        kernel_summary=True,
        short_kernel_study=True,
        group_by_parent_module=True,
        group_by_num_kernels=True,
        include_overlap_info=True,
        topk_ops=10,
        topk_roofline_ops=5,
    )
    assert (out / "gpu_timeline.csv").exists()
    assert "gpu_timeline" in result


def test_inference_report_comparison_and_debug_columns(tmp_path, monkeypatch):
    trace1 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 100)], "t1.json")
    trace2 = _write_trace(tmp_path, [("aten::mm", "gemm_kernel", 120)], "t2.json")
    monkeypatch.setenv("TRACELENS_DEBUG", "1")
    result = generate_inference_report(
        profile_json_path=trace1,
        comparison_json_path=trace2,
        output_csvs_dir=str(tmp_path / "cmp_csvs"),
        output_xlsx_path=str(tmp_path / "cmp.xlsx"),
        include_call_stack=True,
        group_by_parent_module=True,
        collective_analysis=False,
    )
    assert "gpu_timeline" in result
    up = result.get("unified_perf_summary")
    if up is not None and not up.empty and "call_stack_full" in up.columns:
        assert "entry_point" in up.columns


def test_piecewise_capture_merge():
    case_dir = os.path.join(INFERENCE_ROOT, "vllm_prefilldecode_piecewise")
    capture = os.path.join(case_dir, "capture_traces")
    metadata = os.path.join(capture, "execution_details.json")
    graph = os.path.join(case_dir, "graph_execution.json.gz")
    if not all(os.path.isfile(p) for p in (metadata, graph)):
        pytest.skip("piecewise fixture missing")
    merged = merge_capture_trace_into_graph(capture, metadata, graph)
    assert len(merged.events) > 1000


def test_align_streams_multistream_tiebreak():

    graph = [
        {"name": "k1", "args": {"stream": 1}},
        {"name": "k1", "args": {"stream": 2}},
        {"name": "k2", "args": {"stream": 1}},
    ]
    capture = [
        {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        {"name": "hipLaunchKernel", "args": {"kernel": "k1"}},
        {"name": "hipLaunchKernel", "args": {"kernel": "k2"}},
    ]
    aligned = align_streams(graph, capture)
    assert aligned is not None
    assert len(aligned) == 3


def test_verify_subtree_direct_match():
    capture = [{"name": "hipLaunchKernel", "args": {"kernel": "k1"}}]
    graph = [{"name": "k1", "args": {}}]
    code, cap, gr = verify_subtree_events(capture, graph)
    assert code == 1


def test_inference_on_merged_tree(tmp_path):
    case_dir = os.path.join(INFERENCE_ROOT, "vllm_decode_full")
    capture = os.path.join(case_dir, "capture_traces")
    metadata = os.path.join(capture, "execution_details.json")
    graph = os.path.join(case_dir, "graph_execution.json.gz")
    if not all(os.path.isfile(p) for p in (metadata, graph)):
        pytest.skip("fixture missing")
    merged = merge_capture_trace_into_graph(capture, metadata, graph)
    result = generate_inference_report(
        profile_json_path=graph,
        augmented_tree=merged,
        output_csvs_dir=str(tmp_path / "merged_csv"),
        output_xlsx_path=str(tmp_path / "merged.xlsx"),
        collective_analysis=False,
        enable_pseudo_ops=True,
        group_by_parent_module=True,
        kernel_summary=True,
    )
    assert "gpu_timeline" in result


class TestInferenceReportSweep:
    def test_full_flag_matrix(self, tmp_path):
        trace = tmp_path / "trace.json"
        trace.write_text(
            json.dumps(
                _build_synthetic_trace(
                    [
                        ("aten::mm", "gemm_kernel", 100),
                        ("aten::add", "vectorized_elementwise_kernel", 20),
                    ]
                )
            )
        )
        generate_inference_report(
            profile_json_path=str(trace),
            output_csvs_dir=str(tmp_path / "out"),
            output_xlsx_path=str(tmp_path / "r.xlsx"),
            collective_analysis=True,
            kernel_summary=True,
            short_kernel_study=True,
            include_overlap_info=True,
            group_by_parent_module=True,
            group_by_num_kernels=True,
            enable_pseudo_ops=False,
            micro_idle_thresh_us=1,
            topk_ops=10,
            topk_roofline_ops=5,
        )
        assert (tmp_path / "out" / "gpu_timeline.csv").exists()
