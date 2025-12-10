###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import os
import pandas as pd
import tempfile
from TraceLens.util import RocprofParser
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer
from TraceLens.Reporting.generate_perf_report_rocprof import generate_perf_report_rocprof


# Sample rocprof file path (update with actual path)
SAMPLE_ROCPROF_FILE = "/home/vivekag/scratch/assignment/aorta_work/aorta/rocprof_traces/run_20251209_070939/smci350-zts-gtu-c6-25/908_results.json"


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_ROCPROF_FILE),
    reason="Sample rocprof file not found"
)
class TestRocprofParser:
    """Test suite for RocprofParser"""
    
    def test_load_rocprof_data(self):
        """Test loading rocprof JSON file"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
        assert 'rocprofiler-sdk-tool' in data
        assert isinstance(data['rocprofiler-sdk-tool'], list)
        assert len(data['rocprofiler-sdk-tool']) > 0
    
    def test_load_invalid_file(self):
        """Test that invalid files raise an error"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": "data"}')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Not a valid rocprofv3 file"):
                RocprofParser.load_rocprof_data(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_extract_kernel_events(self):
        """Test extracting kernel events"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
        kernel_events = RocprofParser.extract_kernel_events(data)
        
        assert isinstance(kernel_events, list)
        assert len(kernel_events) > 0
        
        # Check first kernel event structure
        first_event = kernel_events[0]
        assert 'name' in first_event
        assert 'ts' in first_event
        assert 'dur' in first_event
        assert 'grid' in first_event
        assert 'block' in first_event
        assert 'stream' in first_event
        
        # Verify grid and block are tuples of 3
        assert isinstance(first_event['grid'], tuple)
        assert len(first_event['grid']) == 3
        assert isinstance(first_event['block'], tuple)
        assert len(first_event['block']) == 3
    
    def test_extract_memory_events(self):
        """Test extracting memory events"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
        memory_events = RocprofParser.extract_memory_events(data)
        
        assert isinstance(memory_events, list)
        # Memory events may be empty
    
    def test_extract_api_events(self):
        """Test extracting API events"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
        api_events = RocprofParser.extract_api_events(data)
        
        assert isinstance(api_events, list)
        # API events may be empty
    
    def test_get_metadata(self):
        """Test extracting metadata"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
        metadata = RocprofParser.get_metadata(data)
        
        assert isinstance(metadata, dict)
        assert 'pid' in metadata
        assert 'init_time' in metadata
        assert 'fini_time' in metadata
        assert 'hostname' in metadata
        assert 'agents' in metadata
        
        # Verify times are reasonable
        assert metadata['fini_time'] > metadata['init_time']


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_ROCPROF_FILE),
    reason="Sample rocprof file not found"
)
class TestRocprofAnalyzer:
    """Test suite for RocprofAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer from sample file"""
        data = RocprofParser.load_rocprof_data(SAMPLE_ROCPROF_FILE)
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
        assert 'type' in df.columns
        assert 'time ms' in df.columns
        assert 'percent' in df.columns
        
        # Check that we have expected row types
        types = df['type'].tolist()
        assert 'total_time' in types
        assert 'kernel' in types
        
        # Percentages should sum to ~100% (excluding total_time row)
        non_total = df[df['type'] != 'total_time']
        total_pct = non_total['percent'].sum()
        assert 99 < total_pct <= 100.1  # Allow for rounding
    
    def test_get_df_kernel_summary(self, analyzer):
        """Test kernel summary generation"""
        df = analyzer.get_df_kernel_summary()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'name' in df.columns
        assert 'Count' in df.columns
        assert 'Total Kernel Time (µs)' in df.columns
        assert 'Mean Kernel Time (µs)' in df.columns
        assert 'Percentage (%)' in df.columns
        
        # Check that data is sorted by total time
        assert df['Total Kernel Time (µs)'].is_monotonic_decreasing
        
        # Percentages should sum to 100%
        assert 99.9 < df['Percentage (%)'].sum() <= 100.1
    
    def test_get_df_kernel_summary_by_category(self, analyzer):
        """Test kernel category summary"""
        df = analyzer.get_df_kernel_summary_by_category()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'op category' in df.columns
        assert 'Count' in df.columns
        assert 'total_direct_kernel_time_ms' in df.columns
        assert 'Percentage (%)' in df.columns
        
        # Check that data is sorted by total time
        assert df['total_direct_kernel_time_ms'].is_monotonic_decreasing
    
    def test_get_df_short_kernels(self, analyzer):
        """Test short kernel analysis"""
        df = analyzer.get_df_short_kernels(threshold_us=10.0)
        
        assert isinstance(df, pd.DataFrame)
        # May be empty if no short kernels
        
        if not df.empty:
            assert 'name' in df.columns
            assert 'Short Kernel count' in df.columns
            assert 'Short Kernel duration (µs) sum' in df.columns
    
    def test_get_df_kernel_details(self, analyzer):
        """Test detailed kernel information"""
        df = analyzer.get_df_kernel_details()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'name' in df.columns
        assert 'Kernel duration (µs)' in df.columns
        assert 'grid_x' in df.columns
        assert 'block_x' in df.columns
        
        # Test topk parameter
        df_top10 = analyzer.get_df_kernel_details(topk=10)
        assert len(df_top10) <= 10


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_ROCPROF_FILE),
    reason="Sample rocprof file not found"
)
class TestGeneratePerfReport:
    """Test suite for end-to-end report generation"""
    
    def test_generate_excel_report(self):
        """Test generating Excel report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_report.xlsx')
            
            dfs = generate_perf_report_rocprof(
                profile_json_path=SAMPLE_ROCPROF_FILE,
                output_xlsx_path=output_path,
                kernel_summary=True,
                kernel_details=False,
                short_kernel_study=False
            )
            
            # Check that file was created
            assert os.path.exists(output_path)
            
            # Check returned dataframes
            assert isinstance(dfs, dict)
            assert 'gpu_timeline' in dfs
            assert 'kernel_summary' in dfs
            assert 'kernel_summary_by_category' in dfs
    
    def test_generate_csv_reports(self):
        """Test generating CSV reports"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dfs = generate_perf_report_rocprof(
                profile_json_path=SAMPLE_ROCPROF_FILE,
                output_csvs_dir=tmpdir,
                kernel_summary=True,
                short_kernel_study=True
            )
            
            # Check that CSV files were created
            csv_files = os.listdir(tmpdir)
            assert len(csv_files) > 0
            assert any('gpu_timeline' in f for f in csv_files)
            assert any('kernel_summary' in f for f in csv_files)
    
    def test_generate_with_all_options(self):
        """Test generating report with all options enabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'full_report.xlsx')
            
            dfs = generate_perf_report_rocprof(
                profile_json_path=SAMPLE_ROCPROF_FILE,
                output_xlsx_path=output_path,
                kernel_summary=True,
                kernel_details=True,
                short_kernel_study=True,
                short_kernel_threshold_us=20,
                topk_kernels=50
            )
            
            assert os.path.exists(output_path)
            assert 'kernel_details' in dfs
            assert 'short_kernels_summary' in dfs
            assert 'short_kernel_histogram' in dfs
            
            # Verify topk worked
            if len(dfs['kernel_details']) > 0:
                assert len(dfs['kernel_details']) <= 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

