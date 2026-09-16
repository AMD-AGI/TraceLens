###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# Regression tests for generate_perf_report_pytorch (inference mode).
# Each test case is a subdirectory under tests/traces/inference/ containing:
#   - A .json.gz trace file
#   - A perf_csvs/ folder with reference CSV files (one per output sheet)
#   - Optionally capture_traces/ (graph capture mode)
#   - Optionally gpu_arch.json

import os, shutil, numpy as np, pandas as pd, pytest, ast, re, gzip, json, glob
from pandas.api.types import is_float_dtype
from TraceLens.Reporting.generate_perf_report_pytorch import (
    classify_graph_capture_trace,
    generate_perf_report_pytorch,
    generate_perf_report_pytorch as gen_inf,
    generate_perf_report_pytorch as generate_inference_report,
)
from TraceLens.TreePerf.tree_perf import TreePerfAnalyzer
import TraceLens.Trace2Tree.trace_capture_merge_experimental as _merge_mod
from TraceLens.Trace2Tree.trace_capture_merge_experimental import (
    _align_capture_to_graph,
    _align_graph_to_capture_by_group,
    _capture_kernel_name,
    _capture_tree_cache,
    _get_cached_capture_tree,
    align_streams,
    build_execution_graph_root_map,
    capture_has_kernel_names,
    get_subtree_events,
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
        metadata_json_path = os.path.join(capture_folder, "execution_details.json")
        classify_graph_capture_trace(capture_folder)
        graph_tree = merge_capture_trace_into_graph(
            capture_folder,
            metadata_json_path,
            profile_path,
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


def test_inference_perf_report_default_output_path(tmp_path):
    """No output_xlsx_path/output_csvs_dir given -> auto-derive next to the
    input trace, replacing the '.json' suffix."""
    cases = find_inference_test_cases()
    if not cases:
        pytest.skip("No inference trace fixtures found")
    dirpath, trace_gz, _capture_folder, _gpu_arch_path = cases[0].values
    profile_path = shutil.copy(os.path.join(dirpath, trace_gz), tmp_path)

    result = generate_perf_report_pytorch(profile_json_path=profile_path)

    expected_xlsx = profile_path.rsplit(".json", 1)[0] + "_perf_report.xlsx"
    assert os.path.exists(expected_xlsx)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Capture merge validation: verifies that kernel timing from the replay trace
# and input args/call stacks from the capture trace are correctly reflected
# in the report for a specific GEMM shape.
# ---------------------------------------------------------------------------

XDIT_TRACE_DIR = os.path.join(INFERENCE_TRACES_ROOT, "xdit_flux.1")
TARGET_GEMM_DIMS = ((768, 15360), (15360, 3072), (768, 3072))


def test_xdit_capture_merge():
    """Capture merge validation for a specific GEMM in the xDiT trace.

    Selects every aten::mm with dims ((4608, 15360), (15360, 3072), (4608, 3072))
    from the capture trace, finds the corresponding kernels in the replay trace
    via positional index, and checks the report's timing and call stacks match.

    Uses the reference CSVs in perf_csvs/ (validated by test_inference_perf_report).
    """

    _merge_mod._capture_tree_cache.clear()

    profile_path = glob.glob(os.path.join(XDIT_TRACE_DIR, "*.json.gz"))[0]
    capture_folder = os.path.join(XDIT_TRACE_DIR, "capture_traces")
    capture_path = glob.glob(os.path.join(capture_folder, "*.json.gz"))[0]
    ref_csvs_dir = os.path.join(XDIT_TRACE_DIR, "perf_csvs")

    # --- 1. Load capture trace: find target GEMMs ---
    key = ("single", os.path.abspath(capture_path))
    cap_tree, cap_roots, cap_root_data = _get_cached_capture_tree(key, capture_path)
    cached_events, filtered_uids = cap_root_data[0]
    UID = _merge_mod.UID
    cap_filtered = [e for e in cached_events if e[UID] in filtered_uids]

    target_dims = [list(d) for d in TARGET_GEMM_DIMS]
    cap_mm_ops = [
        e
        for e in cap_tree.events
        if e.get("name") == "aten::mm"
        and e.get("cat") == "cpu_op"
        and e.get("args", {}).get("Input Dims") == target_dims
    ]

    # --- 2. Load reference CSVs (generated by test_inference_perf_report) ---
    M, K, N = TARGET_GEMM_DIMS[0][0], TARGET_GEMM_DIMS[0][1], TARGET_GEMM_DIMS[1][1]
    df_gemm = pd.read_csv(os.path.join(ref_csvs_dir, "GEMM.csv"))
    match = df_gemm[
        (df_gemm["param: M"].astype(int) == M)
        & (df_gemm["param: K"].astype(int) == K)
        & (df_gemm["param: N"].astype(int) == N)
    ]
    assert (
        len(match) == 1
    ), f"Expected 1 GEMM row for M={M} K={K} N={N}, got {len(match)}"
    row = match.iloc[0]
    report_time_us = float(row["Kernel Time (µs)_sum"])
    report_count = int(row["name_count"])

    # --- 4. Find exact kernels in replay trace via positional index ---
    # The capture root's filtered events align 1:1 with each graph launch's
    # kernel events. We find the indices of our target GEMM's child kernels
    # in the capture root, then use those same indices into each graph
    # launch's replay kernels to sum the exact runtime.
    cap_uid_to_idx = {e[UID]: i for i, e in enumerate(cap_filtered)}

    target_kernel_indices = []
    for mm_op in cap_mm_ops:
        for child_uid in mm_op.get("children", []):
            if child_uid in cap_uid_to_idx:
                target_kernel_indices.append(cap_uid_to_idx[child_uid])

    graph_perf = TreePerfAnalyzer.from_file(profile_path, add_python_func=True)
    exec_map = build_execution_graph_root_map(graph_perf.tree)
    graph_roots = [gl_root for _, g_roots in exec_map for gl_root in g_roots]

    replay_time_us = 0
    replay_count = 0
    for gl_root in graph_roots:
        _, gf = get_subtree_events(
            graph_perf.tree,
            gl_root,
            cat_filter=["kernel", "gpu_memset", "gpu_memcpy"],
        )
        for idx in target_kernel_indices:
            cap_name = _capture_kernel_name(cap_filtered[idx])
            replay_name = gf[idx].get("name", "")
            assert cap_name == replay_name, (
                f"Kernel name mismatch at index {idx}: "
                f"capture='{cap_name}' vs replay='{replay_name}'"
            )
            replay_time_us += gf[idx].get("dur", 0)
            replay_count += 1

    assert (
        replay_count == report_count
    ), f"Kernel count mismatch: replay={replay_count} vs report={report_count}"
    assert abs(report_time_us - replay_time_us) / replay_time_us < 0.001, (
        f"Kernel time mismatch: report={report_time_us:.1f}µs "
        f"vs replay={replay_time_us:.1f}µs"
    )

    # --- 5. Verify non-Synthetic ops have input dims and kernel names in call stacks ---
    df_unified = pd.read_csv(os.path.join(ref_csvs_dir, "unified_perf_summary.csv"))
    non_synth = df_unified[~df_unified["name"].str.contains("Synthetic Op", na=False)]
    empty_dims = non_synth[
        non_synth["Input Dims"].isna() | (non_synth["Input Dims"] == "")
    ]
    assert empty_dims.empty, (
        f"{len(empty_dims)} non-Synthetic ops have empty Input Dims: "
        f"{empty_dims['name'].tolist()[:5]}"
    )
    for idx, row in non_synth.iterrows():
        cs = str(row.get("call_stack_full", ""))
        kd = str(row.get("kernel_details_summary", ""))
        kernel_names = re.findall(r"'name': '([^']+)'", kd)
        for kernel_name in kernel_names:
            assert kernel_name in cs, (
                f"Row {idx} ({row['name']}): kernel '{kernel_name[:60]}' "
                f"not found in call_stack_full"
            )

    # --- 6. Verify call stacks contain _run_timed_pipe and aten::mm ---
    mm_rows = df_unified[df_unified["name"] == "aten::mm"]
    has_expected_stack = mm_rows["call_stack_full"].apply(
        lambda x: isinstance(x, str) and "_run_timed_pipe" in x and "aten::mm" in x
    )
    assert has_expected_stack.all(), (
        f"{has_expected_stack.sum()}/{len(mm_rows)} aten::mm rows "
        f"have _run_timed_pipe and aten::mm in call stack"
    )

    _merge_mod._capture_tree_cache.clear()


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
