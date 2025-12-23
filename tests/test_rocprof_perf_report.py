###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import os
import pandas as pd
import tempfile
from pathlib import Path
from TraceLens.util import RocprofParser
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer
from TraceLens.Reporting.generate_perf_report_rocprof import (
    generate_perf_report_rocprof,
)


def find_test_files(ref_root):
    """
    Recursively find all .json.gz in ref_root.
    Returns a list of file paths.
    """
    test_files = []
    for dirpath, _, filenames in os.walk(ref_root):
        gz_files = [f for f in filenames if f.endswith(".json.gz")]
        for gz in gz_files:
            test_files.append((os.path.join(dirpath, gz)))
    return test_files


@pytest.mark.parametrize("rocprof_file", find_test_files("tests/rocprof"))
class TestRocprofParser:
    """Test suite for RocprofParser"""

    def test_load_rocprof_data(self, rocprof_file):
        """Test loading rocprof JSON file (supports .json and .json.gz)"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        assert "rocprofiler-sdk-tool" in data
        assert isinstance(data["rocprofiler-sdk-tool"], list)
        assert len(data["rocprofiler-sdk-tool"]) > 0

    def test_load_invalid_file(self):
        """Test that invalid files raise an error"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": "data"}')
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Not a valid rocprofv3 file"):
                RocprofParser.load_rocprof_data(temp_file)
        finally:
            os.unlink(temp_file)

    def test_extract_kernel_events(self, rocprof_file):
        """Test extracting kernel events"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        kernel_events = RocprofParser.extract_kernel_events(data)

        assert isinstance(kernel_events, list)
        assert len(kernel_events) > 0

        # Check first kernel event structure
        first_event = kernel_events[0]
        assert "name" in first_event
        assert "ts" in first_event
        assert "dur" in first_event
        assert "grid" in first_event
        assert "block" in first_event
        assert "stream" in first_event

        # Verify grid and block are tuples of 3
        assert isinstance(first_event["grid"], tuple)
        assert len(first_event["grid"]) == 3
        assert isinstance(first_event["block"], tuple)
        assert len(first_event["block"]) == 3

    def test_extract_memory_events(self, rocprof_file):
        """Test extracting memory events"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        memory_events = RocprofParser.extract_memory_events(data)

        assert isinstance(memory_events, list)
        # Memory events may be empty

    def test_extract_api_events(self, rocprof_file):
        """Test extracting API events"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        api_events = RocprofParser.extract_api_events(data)

        assert isinstance(api_events, list)
        # API events may be empty

    def test_get_metadata(self, rocprof_file):
        """Test extracting metadata"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        metadata = RocprofParser.get_metadata(data)

        assert isinstance(metadata, dict)
        assert "pid" in metadata
        assert "init_time" in metadata
        assert "fini_time" in metadata
        assert "hostname" in metadata
        assert "agents" in metadata

        # Verify times are reasonable
        assert metadata["fini_time"] > metadata["init_time"]


@pytest.mark.parametrize("rocprof_file", find_test_files("tests/rocprof"))
class TestRocprofAnalyzer:
    """Test suite for RocprofAnalyzer"""

    @pytest.fixture
    def analyzer(self, rocprof_file):
        """Create analyzer from sample file"""
        data = RocprofParser.load_rocprof_data(rocprof_file)
        kernel_events = RocprofParser.extract_kernel_events(data)
        memory_events = RocprofParser.extract_memory_events(data)
        api_events = RocprofParser.extract_api_events(data)
        metadata = RocprofParser.get_metadata(data)

        return RocprofAnalyzer(kernel_events, memory_events, api_events, metadata)

    def test_get_df_gpu_timeline(self, analyzer):
        """Test GPU timeline generation"""
        df = analyzer.get_df_gpu_timeline()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "type" in df.columns
        assert "time ms" in df.columns
        assert "percent" in df.columns

        # Check that we have expected row types
        types = df["type"].tolist()
        assert "total_time" in types
        assert "kernel" in types
        assert "busy_time" in types  # New: check for busy_time row
        assert "idle" in types

        # Verify that busy_time and idle percentages sum to 100%
        # Note: We don't sum kernel+memory+busy+idle because kernel and memory
        # can overlap, while busy_time is their merged time
        busy_pct = df[df["type"] == "busy_time"]["percent"].values[0]
        idle_pct = df[df["type"] == "idle"]["percent"].values[0]
        assert busy_pct + idle_pct == pytest.approx(100.0, abs=0.1)

        # Verify busy_time + idle = total_time (within rounding)
        busy_time = df[df["type"] == "busy_time"]["time ms"].values[0]
        idle_time = df[df["type"] == "idle"]["time ms"].values[0]
        total_time = df[df["type"] == "total_time"]["time ms"].values[0]
        assert (
            abs((busy_time + idle_time) - total_time) < 0.01
        )  # Allow small rounding error

    def test_get_df_kernel_summary(self, analyzer):
        """Test kernel summary generation"""
        df = analyzer.get_df_kernel_summary()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "name" in df.columns
        assert "Count" in df.columns
        assert "Total Kernel Time (ms)" in df.columns
        assert "Mean Kernel Time (µs)" in df.columns
        assert "Percentage (%)" in df.columns
        assert "Category" in df.columns

        # Verify redundant µs column is removed
        assert "Total Kernel Time (µs)" not in df.columns

        # Check that data is sorted by total time (descending)
        assert df["Total Kernel Time (ms)"].is_monotonic_decreasing

        # Percentages should sum to 100%
        assert 99.9 < df["Percentage (%)"].sum() <= 100.1

    def test_get_df_kernel_summary_by_category(self, analyzer):
        """Test kernel category summary"""
        df = analyzer.get_df_kernel_summary_by_category()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "op category" in df.columns
        assert "Count" in df.columns
        assert "total_direct_kernel_time_ms" in df.columns
        assert "Percentage (%)" in df.columns

        # Check that data is sorted by total time
        assert df["total_direct_kernel_time_ms"].is_monotonic_decreasing

    def test_get_df_short_kernels(self, analyzer):
        """Test short kernel analysis"""
        df = analyzer.get_df_short_kernels(threshold_us=10.0)

        assert isinstance(df, pd.DataFrame)
        # May be empty if no short kernels

        if not df.empty:
            assert "name" in df.columns
            assert "Short Kernel count" in df.columns
            assert "Short Kernel duration (µs) sum" in df.columns

    def test_get_df_kernel_details(self, analyzer):
        """Test detailed kernel information"""
        df = analyzer.get_df_kernel_details()

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "name" in df.columns
        assert "Kernel duration (µs)" in df.columns
        assert "grid_x" in df.columns
        assert "block_x" in df.columns

        # Test topk parameter
        df_top10 = analyzer.get_df_kernel_details(topk=10)
        assert len(df_top10) <= 10


@pytest.mark.parametrize("rocprof_file", find_test_files("tests/rocprof"))
class TestGeneratePerfReport:
    """Test suite for end-to-end report generation"""

    def test_generate_excel_report(self, rocprof_file):
        """Test generating Excel report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.xlsx")

            dfs = generate_perf_report_rocprof(
                profile_json_path=rocprof_file,
                output_xlsx_path=output_path,
                kernel_summary=True,
                kernel_details=False,
                short_kernel_study=False,
            )

            # Check that file was created
            assert os.path.exists(output_path)

            # Check returned dataframes
            assert isinstance(dfs, dict)
            assert "gpu_timeline" in dfs
            assert "kernel_summary" in dfs
            assert "kernel_summary_by_category" in dfs

    def test_generate_csv_reports(self, rocprof_file):
        """Test generating CSV reports"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = generate_perf_report_rocprof(
                profile_json_path=rocprof_file,
                output_csvs_dir=tmpdir,
                kernel_summary=True,
                short_kernel_study=True,
            )

            # Check that CSV files were created
            csv_files = os.listdir(tmpdir)
            assert len(csv_files) > 0
            assert any("gpu_timeline" in f for f in csv_files)
            assert any("kernel_summary" in f for f in csv_files)

    def test_generate_with_all_options(self, rocprof_file):
        """Test generating report with all options enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "full_report.xlsx")

            dfs = generate_perf_report_rocprof(
                profile_json_path=rocprof_file,
                output_xlsx_path=output_path,
                kernel_summary=True,
                kernel_details=True,
                short_kernel_study=True,
                short_kernel_threshold_us=20,
                topk_kernels=50,
            )

            assert os.path.exists(output_path)
            assert "kernel_details" in dfs
            assert "short_kernels_summary" in dfs
            assert "short_kernel_histogram" in dfs

            # Verify topk worked
            if len(dfs["kernel_details"]) > 0:
                assert len(dfs["kernel_details"]) <= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
