###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit and integration tests for the Genesis physics-sim TraceLens extension.

Covers:
  - genesis_analysis: kernel categorization, interval merging, steady-state detection
  - genesis_rocprof_util: CSV-to-JSON conversion, capture loading, benchmark window
  - generate_perf_report_genesis: sheet naming, Excel/MD output, fallback resolution
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TraceLens.Reporting.genesis_analysis import (
    _gpu_timeline_from_intervals,
    _merge_intervals,
    apply_genesis_categories_to_rocprof,
    categorize_kernel,
    compute_genesis_category_summary,
    compute_steady_state_timeline,
    detect_steady_state_cutoff_ns,
    fix_rocprof_kernel_summary_units,
    rebuild_kernel_summary_by_category,
)
from TraceLens.Reporting.genesis_rocprof_util import (
    convert_rocprof_csv_to_json,
    infer_benchmark_window_s,
    load_capture,
    resolve_profile_json,
)
from TraceLens.Reporting.generate_perf_report_genesis import (
    _resolve_steady_state_fallback_s,
    _rocprof_sheets_for_excel,
    write_excel,
    write_genesis_summary_md,
)

###############################################################################
# Shared fixtures — realistic CSV content from actual MI300X genesis traces
###############################################################################

KERNEL_TRACE_CSV = """\
"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH","Agent 2",1,0,70,1,33,"__amd_rocclr_fillBufferAligned",119662,172352210005122,172352210008687,0,0,12,4,48,256,1,1,256,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,2,16,"runtime_get_memory_requirements",119670,172352210061004,172352210062686,0,0,4,4,16,1,1,1,1,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,3,31,"__amd_rocclr_copyBuffer",119696,172352210143326,172352210149335,0,0,16,0,32,512,1,1,512,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,4,1145,"_kernel_solve_body_tiled_wc_amdgpu_c530_0_kernel_0_range_for",567087,172471891336797,172471894892116,8192,352,128,160,112,64,1,1,65536,1,1
"KERNEL_DISPATCH","Agent 2",1,0,70,5,1150,"kernel_step_2_c534_0_kernel_16_range_for",567261,172471895508391,172471895631173,0,340,128,160,112,64,1,1,622592,1,1
"""

HIP_API_TRACE_CSV = """\
"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HIP_COMPILER_API_EXT","__hipRegisterFatBinary",70,70,1,172346145326041,172346145334514
"HIP_RUNTIME_API","hipMalloc",70,70,100,172346150000000,172346150050000
"HIP_RUNTIME_API","hipLaunchKernel",70,70,119662,172352209990000,172352210005000
"""

HSA_API_TRACE_CSV = """\
"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HSA_CORE_API","hsa_signal_wait_scacquire",70,70,1,172352210010000,172352210015000
"""

AGENT_INFO_CSV = """\
"Node_Id","Logical_Node_Id","Agent_Type","Cpu_Cores_Count","Simd_Count","Cpu_Core_Id_Base","Simd_Id_Base","Max_Waves_Per_Simd","Lds_Size_In_Kb","Gds_Size_In_Kb","Num_Gws","Wave_Front_Size","Num_Xcc","Cu_Count","Array_Count","Num_Shader_Banks","Simd_Arrays_Per_Engine","Cu_Per_Simd_Array","Simd_Per_Cu","Max_Slots_Scratch_Cu","Gfx_Target_Version","Vendor_Id","Device_Id","Location_Id","Domain","Drm_Render_Minor","Num_Sdma_Engines","Num_Sdma_Xgmi_Engines","Num_Sdma_Queues_Per_Engine","Num_Cp_Queues","Max_Engine_Clk_Ccompute","Max_Engine_Clk_Fcompute","Sdma_Fw_Version","Fw_Version","Capability","Cu_Per_Engine","Max_Waves_Per_Cu","Family_Id","Workgroup_Max_Size","Grid_Max_Size","Local_Mem_Size","Hive_Id","Gpu_Id","Workgroup_Max_Dim_X","Workgroup_Max_Dim_Y","Workgroup_Max_Dim_Z","Grid_Max_Dim_X","Grid_Max_Dim_Y","Grid_Max_Dim_Z","Name","Vendor_Name","Product_Name","Model_Name"
0,0,"CPU",192,0,0,0,0,0,0,0,0,1,192,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2400,0,0,0,0,0,0,25,0,0,0,0,0,0,0,0,0,0,0,"AMD EPYC 9654 96-Core Processor","CPU","AMD EPYC 9654 96-Core Processor",""
2,2,"GPU",0,1216,0,2147487744,8,64,0,64,64,8,304,32,32,1,10,4,32,90402,4098,29857,1280,0,128,2,14,8,24,2400,2100,24,186,2893521536,9,32,141,1024,4294967295,0,18143416471259055690,28851,1024,1024,1024,2147483647,65535,65535,"gfx942","AMD","AMD Instinct MI300X","ip discovery"
"""


def _create_capture_dir(tmp_dir: Path, include_pftrace: bool = False) -> Path:
    """Create a realistic capture directory layout."""
    capture = tmp_dir / "20260529_181047"
    kernel_trace = capture / "kernel_trace"
    kernel_trace.mkdir(parents=True)

    (kernel_trace / "kernel_kernel_trace.csv").write_text(KERNEL_TRACE_CSV)
    (kernel_trace / "kernel_hip_api_trace.csv").write_text(HIP_API_TRACE_CSV)
    (kernel_trace / "kernel_hsa_api_trace.csv").write_text(HSA_API_TRACE_CSV)
    (kernel_trace / "kernel_agent_info.csv").write_text(AGENT_INFO_CSV)

    if include_pftrace:
        (kernel_trace / "kernel_results.pftrace").write_bytes(b"\x00" * 10)

    return capture


###############################################################################
# genesis_analysis — categorize_kernel
###############################################################################


class TestCategorizeKernel:
    """Validate Genesis physics kernel categorization."""

    def test_rigid_body_solver(self):
        assert (
            categorize_kernel(
                "_kernel_solve_body_tiled_wc_amdgpu_c530_0_kernel_0_range_for"
            )
            == "Rigid Body Solver"
        )

    def test_rigid_body_solver_init(self):
        assert (
            categorize_kernel("func_solve_init_c478_0_kernel_8_range_for")
            == "Rigid Body Solver"
        )

    def test_broadphase_collision(self):
        assert (
            categorize_kernel("func_broad_phase_c402_0_kernel_3_range_for")
            == "Broadphase Collision"
        )

    def test_narrowphase_collision(self):
        assert (
            categorize_kernel("_func_narrowphase_contact0_c422_0_kernel_1_range_for")
            == "Narrowphase Collision"
        )

    def test_narrowphase_multicontact(self):
        assert (
            categorize_kernel(
                "_func_narrowphase_multicontact_mixed_c416_0_kernel_0_range_for"
            )
            == "Narrowphase Collision"
        )

    def test_contact_management(self):
        assert (
            categorize_kernel("func_sort_contacts_c412_0_kernel_1_range_for")
            == "Contact Management"
        )

    def test_time_integration_step1(self):
        assert (
            categorize_kernel("kernel_step_1_c532_0_kernel_6_range_for")
            == "Time Integration"
        )

    def test_time_integration_step2(self):
        assert (
            categorize_kernel("kernel_step_2_c534_0_kernel_16_range_for")
            == "Time Integration"
        )

    def test_constraints(self):
        assert (
            categorize_kernel("add_inequality_constraints_c472_0_kernel_3_range_for")
            == "Constraints"
        )

    def test_memory_ops_copy(self):
        assert categorize_kernel("__amd_rocclr_copyBuffer") == "Memory Ops (ROCm)"

    def test_memory_ops_fill(self):
        assert (
            categorize_kernel("__amd_rocclr_fillBufferAligned") == "Memory Ops (ROCm)"
        )

    def test_runtime_init(self):
        assert categorize_kernel("runtime_initialize") == "Runtime Init"
        assert (
            categorize_kernel("runtime_initialize_rand_states_cuda") == "Runtime Init"
        )

    def test_pytorch_runtime(self):
        name = (
            "void at::native::vectorized_elementwise_kernel<4, "
            "at::native::FillFunctor<float>, std::array<char*, 1ul> >"
        )
        assert categorize_kernel(name) == "PyTorch Runtime"

    def test_geometry_aabb(self):
        assert (
            categorize_kernel("kernel_bit_reduction_into_c260_0_kernel_1_range_for")
            == "Geometry / AABB"
        )

    def test_unknown_kernel(self):
        assert categorize_kernel("some_completely_unknown_kernel") == "Other"

    def test_empty_string(self):
        assert categorize_kernel("") == "Other"


###############################################################################
# genesis_analysis — _merge_intervals
###############################################################################


class TestMergeIntervals:
    """Validate interval merging for GPU busy time computation."""

    def test_non_overlapping(self):
        intervals = [(0, 10), (20, 30), (40, 50)]
        assert _merge_intervals(intervals) == [(0, 10), (20, 30), (40, 50)]

    def test_overlapping(self):
        intervals = [(0, 15), (10, 25), (20, 30)]
        assert _merge_intervals(intervals) == [(0, 30)]

    def test_adjacent(self):
        intervals = [(0, 10), (10, 20)]
        assert _merge_intervals(intervals) == [(0, 20)]

    def test_unsorted_input(self):
        intervals = [(40, 50), (0, 10), (5, 15)]
        assert _merge_intervals(intervals) == [(0, 15), (40, 50)]

    def test_empty_list(self):
        assert _merge_intervals([]) == []

    def test_single_interval(self):
        assert _merge_intervals([(100, 200)]) == [(100, 200)]

    def test_fully_nested(self):
        intervals = [(0, 100), (10, 50), (20, 30)]
        assert _merge_intervals(intervals) == [(0, 100)]


###############################################################################
# genesis_analysis — _gpu_timeline_from_intervals
###############################################################################


class TestGpuTimelineFromIntervals:
    """Validate GPU timeline DataFrame creation."""

    def test_full_busy(self):
        tl = _gpu_timeline_from_intervals(0, 1_000_000, [(0, 1_000_000)])
        assert tl.loc[tl["type"] == "busy_time", "percent"].iloc[0] == pytest.approx(
            100.0
        )
        assert tl.loc[tl["type"] == "idle", "percent"].iloc[0] == pytest.approx(0.0)

    def test_half_busy(self):
        tl = _gpu_timeline_from_intervals(0, 1_000_000, [(0, 500_000)])
        assert tl.loc[tl["type"] == "busy_time", "percent"].iloc[0] == pytest.approx(
            50.0
        )
        assert tl.loc[tl["type"] == "idle", "percent"].iloc[0] == pytest.approx(50.0)

    def test_total_time_ms(self):
        tl = _gpu_timeline_from_intervals(0, 2_000_000, [(0, 1_000_000)])
        total_ms = tl.loc[tl["type"] == "total_time", "time ms"].iloc[0]
        assert total_ms == pytest.approx(2.0)

    def test_columns_present(self):
        tl = _gpu_timeline_from_intervals(0, 100, [(0, 50)])
        assert list(tl.columns) == ["type", "time ms", "percent"]

    def test_expected_row_types(self):
        tl = _gpu_timeline_from_intervals(0, 100, [(0, 50)])
        expected_types = {"total_time", "kernel", "memory", "busy_time", "idle"}
        assert set(tl["type"].tolist()) == expected_types


###############################################################################
# genesis_analysis — detect_steady_state_cutoff_ns
###############################################################################


class TestDetectSteadyStateCutoff:
    """Validate JIT/simulation phase boundary detection."""

    def test_large_gap_detected(self):
        """When there's a large gap (JIT->sim), cutoff should be after the gap."""
        starts = np.array([0, 100, 200, 5_000_000_000, 5_000_001_000, 5_000_002_000])
        ends = np.array([50, 150, 250, 5_000_000_500, 5_000_001_500, 5_000_002_500])
        cutoff, method = detect_steady_state_cutoff_ns(
            starts, ends, gap_threshold_ns=1_000_000_000
        )
        assert cutoff == 5_000_000_000
        assert "after_max_gap" in method

    def test_no_large_gap_uses_fallback(self):
        """When all gaps are small, use the last N seconds of the trace."""
        starts = np.array([0, 100, 200, 300, 400])
        ends = np.array([50, 150, 250, 350, 450])
        cutoff, method = detect_steady_state_cutoff_ns(
            starts, ends, gap_threshold_ns=1_000_000_000, fallback_window_ns=300
        )
        assert "last_" in method
        assert cutoff == 450 - 300

    def test_single_dispatch(self):
        starts = np.array([1000])
        ends = np.array([2000])
        cutoff, method = detect_steady_state_cutoff_ns(starts, ends)
        assert cutoff == 1000
        assert method == "single_dispatch"

    def test_realistic_genesis_pattern(self):
        """Simulate real Genesis trace: sparse init, then dense simulation burst."""
        init_starts = np.array([100, 500, 2000, 10000, 50000])
        init_ends = init_starts + 100
        sim_base = 2_000_000_000
        sim_starts = np.arange(100) * 1_000_000 + sim_base
        sim_ends = sim_starts + 500_000

        all_starts = np.concatenate([init_starts, sim_starts])
        all_ends = np.concatenate([init_ends, sim_ends])

        cutoff, method = detect_steady_state_cutoff_ns(
            all_starts, all_ends, gap_threshold_ns=1_000_000_000
        )
        assert cutoff == sim_base
        assert "after_max_gap" in method


###############################################################################
# genesis_analysis — compute_steady_state_timeline
###############################################################################


class TestComputeSteadyStateTimeline:
    """End-to-end steady-state timeline from a CSV file."""

    def _write_trace_csv(self, tmp_dir: Path, rows: list) -> Path:
        csv_path = tmp_dir / "kernel_kernel_trace.csv"
        header = (
            '"Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id",'
            '"Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id",'
            '"Start_Timestamp","End_Timestamp","LDS_Block_Size",'
            '"Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count",'
            '"Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z",'
            '"Grid_Size_X","Grid_Size_Y","Grid_Size_Z"\n'
        )
        lines = [header]
        for i, (start, end, name) in enumerate(rows):
            lines.append(
                f'"KERNEL_DISPATCH","Agent 2",1,0,70,{i+1},1,"{name}",{1000+i},'
                f"{start},{end},0,0,12,4,48,256,1,1,256,1,1\n"
            )
        csv_path.write_text("".join(lines))
        return csv_path

    def test_basic_two_phase(self):
        """JIT phase then simulation burst -> timeline covers only sim."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            jit_rows = [(1000, 2000, "__amd_rocclr_copyBuffer")]
            sim_base = 3_000_000_000
            sim_rows = [
                (
                    sim_base + i * 10000,
                    sim_base + i * 10000 + 5000,
                    "kernel_step_1_c532_0_kernel_6_range_for",
                )
                for i in range(100)
            ]
            csv_path = self._write_trace_csv(tmp_dir, jit_rows + sim_rows)

            timeline, meta = compute_steady_state_timeline(
                str(csv_path),
                gap_threshold_ns=1_000_000_000,
                fallback_window_ns=500_000_000,
            )

            assert meta["dispatch_count"] == 100
            assert meta["gpu_util_pct"] > 0
            assert "type" in timeline.columns

    def test_returns_dataframe_with_expected_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rows = [
                (
                    1_000_000_000 + i * 1000,
                    1_000_000_000 + i * 1000 + 500,
                    "kernel_step_1",
                )
                for i in range(50)
            ]
            csv_path = self._write_trace_csv(tmp_dir, rows)
            timeline, meta = compute_steady_state_timeline(str(csv_path))

            assert isinstance(timeline, pd.DataFrame)
            assert isinstance(meta, dict)
            assert "method" in meta
            assert "window_ms" in meta


###############################################################################
# genesis_analysis — fix_rocprof_kernel_summary_units
###############################################################################


class TestFixRocprofKernelSummaryUnits:
    """TraceLens reports ns as ms — this function divides by 1000."""

    def test_divides_total_and_mean(self):
        df = pd.DataFrame(
            {
                "name": ["kernel_a", "kernel_b"],
                "Total Kernel Time (ms)": [5000.0, 3000.0],
                "Mean Kernel Time (µs)": [500.0, 300.0],
                "Count": [10, 10],
            }
        )
        fixed = fix_rocprof_kernel_summary_units(df)
        assert fixed["Total Kernel Time (ms)"].iloc[0] == pytest.approx(5.0)
        assert fixed["Total Kernel Time (ms)"].iloc[1] == pytest.approx(3.0)
        assert fixed["Mean Kernel Time (µs)"].iloc[0] == pytest.approx(0.5)

    def test_does_not_modify_original(self):
        df = pd.DataFrame(
            {
                "name": ["k"],
                "Total Kernel Time (ms)": [1000.0],
                "Mean Kernel Time (µs)": [100.0],
            }
        )
        fix_rocprof_kernel_summary_units(df)
        assert df["Total Kernel Time (ms)"].iloc[0] == 1000.0

    def test_missing_columns_no_error(self):
        df = pd.DataFrame({"name": ["k"], "Count": [5]})
        result = fix_rocprof_kernel_summary_units(df)
        assert "Count" in result.columns


###############################################################################
# genesis_analysis — apply_genesis_categories_to_rocprof
###############################################################################


class TestApplyGenesisCategories:
    """Validate in-place category assignment to rocprof reports dict."""

    def _sample_kernel_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "name": [
                    "_kernel_solve_body_tiled_wc_amdgpu_c530_0_kernel_0_range_for",
                    "func_broad_phase_c402_0_kernel_3_range_for",
                    "kernel_step_1_c532_0_kernel_6_range_for",
                    "__amd_rocclr_copyBuffer",
                ],
                "Count": [471, 511, 511, 11723],
                "Total Kernel Time (ms)": [2021.9, 159.1, 159.4, 64.2],
                "Percentage (%)": [50.3, 3.96, 3.97, 1.6],
            }
        )

    def test_adds_category_column(self):
        reports = {"kernel_summary": self._sample_kernel_summary()}
        apply_genesis_categories_to_rocprof(reports)
        assert "Category" in reports["kernel_summary"].columns

    def test_categories_are_correct(self):
        reports = {"kernel_summary": self._sample_kernel_summary()}
        apply_genesis_categories_to_rocprof(reports)
        cats = reports["kernel_summary"]["Category"].tolist()
        assert cats[0] == "Rigid Body Solver"
        assert cats[1] == "Broadphase Collision"
        assert cats[2] == "Time Integration"
        assert cats[3] == "Memory Ops (ROCm)"

    def test_creates_category_summary_sheet(self):
        reports = {"kernel_summary": self._sample_kernel_summary()}
        apply_genesis_categories_to_rocprof(reports)
        assert "kernel_summary_by_category" in reports
        by_cat = reports["kernel_summary_by_category"]
        assert "op category" in by_cat.columns
        assert "Percentage (%)" in by_cat.columns

    def test_handles_empty_summary(self):
        reports = {"kernel_summary": pd.DataFrame()}
        apply_genesis_categories_to_rocprof(reports)
        assert "kernel_summary_by_category" not in reports

    def test_handles_none_summary(self):
        reports = {"kernel_summary": None}
        apply_genesis_categories_to_rocprof(reports)
        assert "kernel_summary_by_category" not in reports


###############################################################################
# genesis_analysis — rebuild_kernel_summary_by_category
###############################################################################


class TestRebuildKernelSummaryByCategory:
    """Validate category grouping aggregation."""

    def test_groups_kernels(self):
        df = pd.DataFrame(
            {
                "name": [
                    "_kernel_solve_body_tiled",
                    "func_solve_init_c478",
                    "func_broad_phase_c402",
                ],
                "Count": [100, 200, 150],
                "Total Kernel Time (ms)": [500.0, 300.0, 200.0],
            }
        )
        df["Category"] = df["name"].apply(categorize_kernel)
        result = rebuild_kernel_summary_by_category(df)

        assert "Rigid Body Solver" in result["op category"].values
        assert "Broadphase Collision" in result["op category"].values
        rbs = result[result["op category"] == "Rigid Body Solver"]
        assert rbs["Count"].iloc[0] == 300

    def test_percentages_sum_to_100(self):
        df = pd.DataFrame(
            {
                "name": ["kernel_step_1", "func_broad_phase"],
                "Count": [10, 20],
                "Total Kernel Time (ms)": [60.0, 40.0],
                "Category": ["Time Integration", "Broadphase Collision"],
            }
        )
        result = rebuild_kernel_summary_by_category(df)
        assert result["Percentage (%)"].sum() == pytest.approx(100.0)

    def test_empty_input(self):
        result = rebuild_kernel_summary_by_category(pd.DataFrame())
        assert result.empty

    def test_none_input(self):
        result = rebuild_kernel_summary_by_category(None)
        assert result.empty


###############################################################################
# genesis_analysis — compute_genesis_category_summary
###############################################################################


class TestComputeGenesisCategorySummary:
    """Validate category summary built from kernel_kernel_stats.csv format."""

    def test_from_realistic_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats_csv = Path(tmp) / "kernel_kernel_stats.csv"
            stats_csv.write_text(
                '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"\n'
                '"_kernel_solve_body_tiled_wc_amdgpu_c530_0_kernel_0_range_for",471,2021965400,4292920,50.33,14862,6725986,1264097\n'
                '"func_broad_phase_c402_0_kernel_3_range_for",511,159149417,311447,3.96,260547,390419,21757\n'
                '"kernel_step_1_c532_0_kernel_6_range_for",511,159422609,311981,3.97,309660,323761,1358\n'
                '"__amd_rocclr_copyBuffer",11723,64186484,5475,1.60,1882,531349,22508\n'
            )
            summary = compute_genesis_category_summary(str(stats_csv))

            assert "Genesis_Category" in summary.columns
            assert "Total_ms" in summary.columns
            assert "Pct" in summary.columns
            assert summary.iloc[0]["Genesis_Category"] == "Rigid Body Solver"

    def test_pct_sums_to_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            stats_csv = Path(tmp) / "stats.csv"
            stats_csv.write_text(
                '"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"\n'
                '"kernel_step_1",100,1000000,10000,50.0,5000,15000,1000\n'
                '"func_broad_phase",100,1000000,10000,50.0,5000,15000,1000\n'
            )
            summary = compute_genesis_category_summary(str(stats_csv))
            assert summary["Pct"].sum() == pytest.approx(100.0)


###############################################################################
# genesis_rocprof_util — convert_rocprof_csv_to_json
###############################################################################


class TestConvertRocprofCsvToJson:
    """Validate CSV-to-JSON conversion for TraceLens rocprof parser."""

    @pytest.fixture()
    def converted_json(self, tmp_path: Path) -> dict:
        """Convert fixture CSVs to JSON once, share across tests."""
        capture = _create_capture_dir(tmp_path)
        out_path = tmp_path / "result.json"
        convert_rocprof_csv_to_json(str(capture / "kernel_trace"), str(out_path))
        return json.loads(out_path.read_text())

    @pytest.fixture()
    def converted_json_with_api(self, tmp_path: Path) -> dict:
        """Convert with include_api=True."""
        capture = _create_capture_dir(tmp_path)
        out_path = tmp_path / "result.json"
        convert_rocprof_csv_to_json(
            str(capture / "kernel_trace"), str(out_path), include_api=True
        )
        return json.loads(out_path.read_text())

    def test_produces_valid_json(self, converted_json: dict):
        assert "rocprofiler-sdk-tool" in converted_json

    def test_has_kernel_dispatches(self, converted_json: dict):
        dispatches = converted_json["rocprofiler-sdk-tool"][0]["buffer_records"][
            "kernel_dispatch"
        ]
        assert len(dispatches) == 5

    def test_dispatch_has_required_fields(self, converted_json: dict):
        dispatch = converted_json["rocprofiler-sdk-tool"][0]["buffer_records"][
            "kernel_dispatch"
        ][0]
        assert "start_timestamp" in dispatch
        assert "end_timestamp" in dispatch
        assert "dispatch_info" in dispatch
        assert "kernel_id" in dispatch["dispatch_info"]

    def test_kernel_symbols_populated(self, converted_json: dict):
        symbols = converted_json["rocprofiler-sdk-tool"][0]["kernel_symbols"]
        assert len(symbols) > 0
        names = [s["kernel_name"] for s in symbols]
        assert "__amd_rocclr_fillBufferAligned" in names

    def test_include_api_false_no_hip_events(self, converted_json: dict):
        assert (
            converted_json["rocprofiler-sdk-tool"][0]["buffer_records"]["hip_api"] == []
        )

    def test_include_api_true_has_hip_events(self, converted_json_with_api: dict):
        hip_events = converted_json_with_api["rocprofiler-sdk-tool"][0][
            "buffer_records"
        ]["hip_api"]
        assert len(hip_events) == 3
        assert hip_events[0]["operation"] == "__hipRegisterFatBinary"

    def test_include_api_true_has_hsa_events(self, converted_json_with_api: dict):
        hsa_events = converted_json_with_api["rocprofiler-sdk-tool"][0][
            "buffer_records"
        ]["hsa_api"]
        assert len(hsa_events) == 1
        assert hsa_events[0]["operation"] == "hsa_signal_wait_scacquire"

    def test_agents_from_agent_info(self, converted_json: dict):
        agents = converted_json["rocprofiler-sdk-tool"][0]["agents"]
        assert len(agents) == 2
        gpu_agent = [a for a in agents if a["type"] == "GPU"]
        assert len(gpu_agent) == 1
        assert gpu_agent[0]["product_name"] == "AMD Instinct MI300X"

    def test_missing_kernel_csv_raises(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            convert_rocprof_csv_to_json(str(empty_dir), str(tmp_path / "out.json"))

    def test_grid_and_workgroup_sizes(self, converted_json: dict):
        solver_dispatch = converted_json["rocprofiler-sdk-tool"][0]["buffer_records"][
            "kernel_dispatch"
        ][3]
        assert solver_dispatch["dispatch_info"]["grid_size"]["x"] == 65536
        assert solver_dispatch["dispatch_info"]["workgroup_size"]["x"] == 64


###############################################################################
# genesis_rocprof_util — infer_benchmark_window_s
###############################################################################


class TestInferBenchmarkWindow:
    """Validate parsing of benchmark wall_time from run.log."""

    def test_parses_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp)
            (capture_dir / "run.log").write_text(
                "n_envs= 8192  |  wall_time=3.98s  |  FPS=125.6  |  throughput=1029427 env·steps/s\n"
            )
            result = infer_benchmark_window_s(capture_dir)
            assert result == pytest.approx(3.98 * 1.05)

    def test_no_run_log_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = infer_benchmark_window_s(Path(tmp))
            assert result is None

    def test_run_log_without_wall_time_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp)
            (capture_dir / "run.log").write_text("some unrelated log output\n")
            result = infer_benchmark_window_s(capture_dir)
            assert result is None

    def test_large_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp)
            (capture_dir / "run.log").write_text("wall_time=120.50s\n")
            result = infer_benchmark_window_s(capture_dir)
            assert result == pytest.approx(120.50 * 1.05)


###############################################################################
# genesis_rocprof_util — load_capture
###############################################################################


class TestLoadCapture:
    """Validate capture directory discovery logic."""

    def test_finds_kernel_trace_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            result = load_capture(str(capture))
            assert result["has_kernel_csv"] is True
            assert result["rocprof_dir"] == capture / "kernel_trace"
            assert result["capture_dir"] == capture.resolve()

    def test_finds_pftrace(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp), include_pftrace=True)
            result = load_capture(str(capture))
            assert result["pftrace"] is not None
            assert "pftrace" in str(result["pftrace"])

    def test_no_pftrace_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp), include_pftrace=False)
            result = load_capture(str(capture))
            assert result["pftrace"] is None

    def test_reads_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            manifest = {
                "timestamp": "20260529_181047",
                "n_envs": 8192,
                "num_steps": 500,
                "precision": "32",
            }
            (capture / "combined_manifest.json").write_text(json.dumps(manifest))
            result = load_capture(str(capture))
            assert result["manifest"]["n_envs"] == 8192
            assert result["manifest"]["precision"] == "32"

    def test_no_manifest_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            result = load_capture(str(capture))
            assert result["manifest"] is None

    def test_viztracer_json_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            (capture / "viztracer_trace.json").write_text("{}")
            result = load_capture(str(capture))
            assert result["viztracer_json"] is not None

    def test_viztracer_json_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            result = load_capture(str(capture))
            assert result["viztracer_json"] is None


###############################################################################
# genesis_rocprof_util — resolve_profile_json
###############################################################################


class TestResolveProfileJson:
    """Validate JSON resolution (prefers CSV->JSON over native rocprof JSON)."""

    def test_generates_json_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            capture_dict = load_capture(str(capture))
            output_dir = Path(tmp) / "work"
            output_dir.mkdir()
            result = resolve_profile_json(capture_dict, output_dir, include_api=False)
            assert result.exists()
            data = json.loads(result.read_text())
            assert "rocprofiler-sdk-tool" in data

    def test_reuses_existing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = _create_capture_dir(Path(tmp))
            capture_dict = load_capture(str(capture))
            output_dir = Path(tmp) / "work"
            output_dir.mkdir()
            resolve_profile_json(capture_dict, output_dir, include_api=False)
            json_path = output_dir / "kernel_results.json"
            original_size = json_path.stat().st_size
            result = resolve_profile_json(capture_dict, output_dir, include_api=False)
            assert result.stat().st_size == original_size

    def test_raises_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_capture = Path(tmp) / "empty_capture"
            empty_capture.mkdir()
            capture_dict = {
                "has_kernel_csv": False,
                "kernel_csv_dir": empty_capture,
                "profile_json": None,
                "capture_dir": empty_capture,
                "rocprof_dir": empty_capture,
            }
            output_dir = Path(tmp) / "work"
            output_dir.mkdir()
            with pytest.raises(FileNotFoundError):
                resolve_profile_json(capture_dict, output_dir, include_api=False)


###############################################################################
# generate_perf_report_genesis — _rocprof_sheets_for_excel
###############################################################################


class TestRocprofSheetsForExcel:
    """Validate filtering of rocprof sheets for Excel output."""

    def test_keeps_known_sheets(self):
        rocprof = {
            "gpu_timeline": pd.DataFrame({"type": ["total"], "time ms": [100]}),
            "kernel_summary": pd.DataFrame({"name": ["k1"]}),
            "kernel_summary_by_category": pd.DataFrame({"op category": ["Solver"]}),
            "internal_debug_thing": pd.DataFrame({"x": [1]}),
        }
        filtered = _rocprof_sheets_for_excel(rocprof)
        assert "gpu_timeline" in filtered
        assert "kernel_summary" in filtered
        assert "kernel_summary_by_category" in filtered
        assert "internal_debug_thing" not in filtered

    def test_skips_empty_dataframes(self):
        rocprof = {
            "gpu_timeline": pd.DataFrame(),
            "kernel_summary": pd.DataFrame({"name": ["k1"]}),
        }
        filtered = _rocprof_sheets_for_excel(rocprof)
        assert "gpu_timeline" not in filtered
        assert "kernel_summary" in filtered

    def test_skips_none_values(self):
        rocprof = {
            "gpu_timeline": None,
            "kernel_summary": pd.DataFrame({"name": ["k1"]}),
        }
        filtered = _rocprof_sheets_for_excel(rocprof)
        assert "gpu_timeline" not in filtered


###############################################################################
# generate_perf_report_genesis — write_excel
###############################################################################


class TestWriteExcel:
    """Validate Excel file generation."""

    def test_creates_xlsx_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "report.xlsx"
            sections = {
                "rocprof": {
                    "gpu_timeline": pd.DataFrame(
                        {
                            "type": ["total_time", "kernel", "idle"],
                            "time ms": [100.0, 80.0, 20.0],
                            "percent": [100.0, 80.0, 20.0],
                        }
                    ),
                },
            }
            write_excel(xlsx_path, sections)
            assert xlsx_path.exists()
            assert xlsx_path.stat().st_size > 0

    def test_multiple_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "report.xlsx"
            sections = {
                "rocprof": {
                    "kernel_summary": pd.DataFrame({"name": ["k1"], "Count": [10]}),
                },
                "pftrace_hip": {
                    "hip_summary": pd.DataFrame({"api": ["hipMalloc"], "count": [5]}),
                },
            }
            write_excel(xlsx_path, sections)
            xl = pd.ExcelFile(xlsx_path)
            assert "kernel_summary" in xl.sheet_names
            assert "pftrace_hip_hip_summary" in xl.sheet_names

    def test_skips_empty_dataframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            xlsx_path = Path(tmp) / "report.xlsx"
            sections = {
                "rocprof": {
                    "gpu_timeline": pd.DataFrame({"x": [1]}),
                    "empty_sheet": pd.DataFrame(),
                },
            }
            write_excel(xlsx_path, sections)
            xl = pd.ExcelFile(xlsx_path)
            assert "empty_sheet" not in xl.sheet_names


###############################################################################
# generate_perf_report_genesis — write_genesis_summary_md
###############################################################################


class TestWriteGenesisSummaryMd:
    """Validate markdown summary generation."""

    def _sample_reports(self) -> dict:
        return {
            "rocprof": {
                "gpu_timeline": pd.DataFrame(
                    {
                        "type": ["total_time", "kernel", "memory", "busy_time", "idle"],
                        "time ms": [3980.0, 3800.0, 0.0, 3800.0, 180.0],
                        "percent": [100.0, 95.5, 0.0, 95.5, 4.5],
                    }
                ),
                "kernel_summary": pd.DataFrame(
                    {
                        "name": [
                            "_kernel_solve_body_tiled_wc_amdgpu_c530_0_kernel_0_range_for",
                            "func_broad_phase_c402_0_kernel_3_range_for",
                        ],
                        "Count": [471, 511],
                        "Total Kernel Time (ms)": [2021.9, 159.1],
                        "Percentage (%)": [50.3, 3.96],
                        "Category": ["Rigid Body Solver", "Broadphase Collision"],
                    }
                ),
                "kernel_summary_by_category": pd.DataFrame(
                    {
                        "op category": ["Rigid Body Solver", "Broadphase Collision"],
                        "Count": [471, 511],
                        "total_direct_kernel_time_ms": [2021.9, 159.1],
                        "Percentage (%)": [92.7, 7.3],
                        "Cumulative Percentage (%)": [92.7, 100.0],
                    }
                ),
            },
        }

    def test_creates_md_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            capture = {"capture_dir": Path(tmp), "manifest": None}
            steady_meta = {
                "method": "after_max_gap_5742.1ms",
                "dispatch_count": 53000,
                "gpu_util_pct": 95.5,
                "window_ms": 3980.0,
            }
            write_genesis_summary_md(
                md_path, capture, self._sample_reports(), steady_meta
            )
            assert md_path.exists()
            assert "TraceLens Genesis Performance Report" in md_path.read_text()

    def test_includes_gpu_utilization(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            capture = {"capture_dir": Path(tmp), "manifest": None}
            steady_meta = {
                "method": "after_max_gap",
                "dispatch_count": 53000,
                "gpu_util_pct": 95.5,
                "window_ms": 3980.0,
            }
            write_genesis_summary_md(
                md_path, capture, self._sample_reports(), steady_meta
            )
            content = md_path.read_text()
            assert "95.5%" in content
            assert "53,000 dispatches" in content

    def test_includes_top_kernels(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            capture = {"capture_dir": Path(tmp), "manifest": None}
            steady_meta = {
                "method": "x",
                "dispatch_count": 1,
                "gpu_util_pct": 50,
                "window_ms": 1,
            }
            write_genesis_summary_md(
                md_path, capture, self._sample_reports(), steady_meta
            )
            content = md_path.read_text()
            assert "Top 10 Kernels" in content
            assert "Rigid Body Solver" in content

    def test_includes_manifest_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            capture = {
                "capture_dir": Path(tmp),
                "manifest": {
                    "timestamp": "20260529_181047",
                    "n_envs": 8192,
                    "num_steps": 500,
                    "precision": "32",
                },
            }
            steady_meta = {
                "method": "x",
                "dispatch_count": 1,
                "gpu_util_pct": 50,
                "window_ms": 1,
            }
            write_genesis_summary_md(
                md_path, capture, self._sample_reports(), steady_meta
            )
            content = md_path.read_text()
            assert "n_envs=8192" in content
            assert "steps=500" in content
            assert "fp32" in content

    def test_handles_empty_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "summary.md"
            capture = {"capture_dir": Path(tmp), "manifest": None}
            write_genesis_summary_md(md_path, capture, {"rocprof": {}}, {})
            assert md_path.exists()
            assert "TraceLens Genesis Performance Report" in md_path.read_text()


###############################################################################
# generate_perf_report_genesis — _resolve_steady_state_fallback_s
###############################################################################


class TestResolveSteadyStateFallback:
    """Validate fallback window resolution (CLI > run.log > default)."""

    def test_cli_value_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp)
            (capture_dir / "run.log").write_text("wall_time=10.0s\n")
            capture = {"capture_dir": capture_dir}
            result = _resolve_steady_state_fallback_s(capture, cli_value=7.0)
            assert result == 7.0

    def test_infers_from_run_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_dir = Path(tmp)
            (capture_dir / "run.log").write_text("wall_time=3.98s\n")
            capture = {"capture_dir": capture_dir}
            result = _resolve_steady_state_fallback_s(capture, cli_value=None)
            assert result == pytest.approx(3.98 * 1.05)

    def test_default_when_no_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = {"capture_dir": Path(tmp)}
            result = _resolve_steady_state_fallback_s(capture, cli_value=None)
            assert result == 5.0
