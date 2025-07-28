#!/usr/bin/env python3
"""
Unit tests for JAX integration using pytest
"""

import pytest
import sys
import os
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree, JaxTreePerfAnalyzer
from TraceLens.TreePerf.jax_analyses import JaxAnalyses
from TraceLens.util import DataLoader, TraceEventUtils


class TestJaxAnalyses:
    """Test JAX analysis functionality"""
    
    def test_event_categorizer_cpu_op(self):
        """Test categorizer correctly identifies CPU operations"""
        # Mock metadata for Framework Name Scope thread
        metadata = {
            4: {
                3735928561: {'thread_name': 'Framework Name Scope'}
            }
        }
        
        # CPU operation event
        cpu_event = {
            'name': 'jit(train_step)',
            'pid': 4,
            'tid': 3735928561,
            'ph': 'X'
        }
        
        result = JaxAnalyses.get_event_category(metadata, cpu_event)
        assert result == "cpu_op"
    
    def test_event_categorizer_kernel(self):
        """Test categorizer correctly identifies kernel events"""
        # Mock metadata for Stream thread
        metadata = {
            1: {
                1: {'thread_name': 'Stream #1'}
            }
        }
        
        # Kernel event
        kernel_event = {
            'name': 'some_kernel_name',
            'pid': 1,
            'tid': 1,
            'ph': 'X'
        }
        
        result = JaxAnalyses.get_event_category(metadata, kernel_event)
        assert result == "kernel"
    
    def test_event_categorizer_synthetic_runtime(self):
        """Test categorizer handles synthetic runtime events"""
        metadata = {}
        
        # Synthetic runtime event
        runtime_event = {
            'name': 'cudaLaunchKernel',
            'cat': 'cuda_runtime',
            'ph': 'X'
        }
        
        result = JaxAnalyses.get_event_category(metadata, runtime_event)
        assert result == "cuda_runtime"
    
    def test_event_categorizer_unknown(self):
        """Test categorizer returns Unknown for unrecognized events"""
        metadata = {}
        
        # Unknown event
        unknown_event = {
            'name': 'unknown_event',
            'pid': 999,
            'tid': 999,
            'ph': 'X'
        }
        
        result = JaxAnalyses.get_event_category(metadata, unknown_event)
        assert result == "Unknown"
    
    def test_prepare_event_categorizer(self):
        """Test categorizer preparation from events"""
        events = [
            {'ph': 'M', 'name': 'thread_name', 'pid': 1, 'tid': 1, 'args': {'name': 'Stream #1'}},
            {'name': 'kernel', 'pid': 1, 'tid': 1, 'ph': 'X'}
        ]
        
        categorizer = JaxAnalyses.prepare_event_categorizer(events)
        
        # Test the categorizer function
        kernel_event = {'name': 'test_kernel', 'pid': 1, 'tid': 1, 'ph': 'X'}
        result = categorizer(kernel_event)
        assert result == "kernel"


class TestJaxXplaneToTree:
    """Test JAX xplane to tree conversion"""
    
    def test_dummy_values(self):
        """Test dummy values are properly defined"""
        dummy_values = JaxXplaneToTree.DUMMY_VALUES
        
        assert 'correlation' in dummy_values
        assert 'External id' in dummy_values
        assert 'Input Dims' in dummy_values
        
        # Check they are recognizable as dummy values
        assert dummy_values['correlation'] < 0
        assert dummy_values['External id'] < 0
        assert 'DUMMY' in dummy_values['Input Dims']
    
    def test_add_pytorch_fields_cpu_op(self):
        """Test adding PyTorch fields to CPU operation"""
        event = {
            'name': 'jit(test)',
            'ph': 'X',
            'args': {}
        }
        
        enhanced = JaxXplaneToTree._add_pytorch_fields(event, 'cpu_op')
        
        assert 'Input Dims' in enhanced['args']
        assert 'correlation' in enhanced['args']
        assert enhanced['args']['correlation'] == JaxXplaneToTree.DUMMY_VALUES['correlation']
    
    def test_add_pytorch_fields_kernel(self):
        """Test adding PyTorch fields to kernel event"""
        event = {
            'name': 'test_kernel',
            'ph': 'X',
            'args': {}
        }
        
        enhanced = JaxXplaneToTree._add_pytorch_fields(event, 'kernel')
        
        assert 'stream' in enhanced['args']
        assert 'correlation' in enhanced['args']
        assert enhanced['args']['stream'] == JaxXplaneToTree.DUMMY_VALUES['stream']


class TestTreeConstruction:
    """Test tree construction logic"""
    
    def test_create_pytorch_compatible_tree_structure_empty(self):
        """Test tree construction with empty input"""
        events = []
        categorizer = lambda x: "unknown"
        
        result = JaxXplaneToTree._create_pytorch_compatible_tree_structure(events, categorizer)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_create_pytorch_compatible_tree_structure_basic(self):
        """Test tree construction with basic events"""
        # Mock events
        cpu_op = {
            'name': 'jit(test)',
            'pid': 4,
            'tid': 3735928561,
            'ts': 1000,
            'dur': 100,
            'ph': 'X',
            'args': {}
        }
        
        kernel = {
            'name': 'test_kernel',
            'pid': 1,
            'tid': 1,
            'ts': 1010,
            'dur': 50,
            'ph': 'X',
            'args': {'hlo_op': 'test.1'}
        }
        
        events = [cpu_op, kernel]
        
        def mock_categorizer(event):
            if event['name'] == 'jit(test)':
                return 'cpu_op'
            elif event['name'] == 'test_kernel':
                return 'kernel'
            return 'unknown'
        
        result = JaxXplaneToTree._create_pytorch_compatible_tree_structure(events, mock_categorizer)
        
        # Should have: 1 other event + 1 CPU op + 1 runtime + 1 kernel = 4 events
        assert len(result) >= 3  # At least CPU op, runtime, kernel
        
        # Check that we have enhanced events
        cpu_ops = [e for e in result if e.get('name') == 'jit(test)']
        assert len(cpu_ops) == 1
        
        # CPU op should have enhanced args
        cpu_op_enhanced = cpu_ops[0]
        assert 'Input Dims' in cpu_op_enhanced['args']
        
        # Should have children (runtime events)
        assert 'children' in cpu_op_enhanced
        assert len(cpu_op_enhanced['children']) > 0


class TestIntegrationBasic:
    """Basic integration tests that don't require large trace files"""
    
    def test_jax_tree_perf_analyzer_creation(self):
        """Test that JaxTreePerfAnalyzer can be created (without actual file)"""
        # This test would need a real xplane.pb file, so we'll just test the class exists
        assert hasattr(JaxTreePerfAnalyzer, 'from_xplane_pb')
        assert callable(JaxTreePerfAnalyzer.from_xplane_pb)
    
    def test_jax_xplane_to_tree_creation(self):
        """Test that JaxXplaneToTree can be created (without actual file)"""
        assert hasattr(JaxXplaneToTree, 'from_xplane_pb')
        assert callable(JaxXplaneToTree.from_xplane_pb)


@pytest.mark.integration
class TestJaxIntegrationWithFile:
    """Integration tests that require the actual trace file"""
    
    @pytest.fixture
    def trace_file_path(self):
        """Fixture providing the trace file path"""
        return "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    
    @pytest.fixture
    def skip_if_no_trace_file(self, trace_file_path):
        """Skip tests if trace file doesn't exist"""
        if not os.path.exists(trace_file_path):
            pytest.skip(f"Trace file not found: {trace_file_path}")
    
    def test_load_trace_data(self, trace_file_path, skip_if_no_trace_file):
        """Test that we can load the trace data"""
        data = DataLoader.load_data(trace_file_path)
        
        assert 'traceEvents' in data
        assert len(data['traceEvents']) > 0
        
        events = data['traceEvents']
        assert len(events) > 1000000  # Should be a large trace
    
    def test_categorizer_creation(self, trace_file_path, skip_if_no_trace_file):
        """Test categorizer creation with real data"""
        data = DataLoader.load_data(trace_file_path)
        events = data['traceEvents']
        
        categorizer = JaxAnalyses.prepare_event_categorizer(events)
        
        # Test with a sample event
        non_metadata_events = TraceEventUtils.non_metadata_events(events)
        if non_metadata_events:
            sample_event = non_metadata_events[0]
            result = categorizer(sample_event)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_tree_construction_performance(self, trace_file_path, skip_if_no_trace_file):
        """Test that tree construction completes in reasonable time"""
        import time
        
        start_time = time.time()
        perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(trace_file_path)
        end_time = time.time()
        
        # Should complete in under 90 seconds (optimized version)
        # Note: Most time is in underlying TraceLens tree construction, not our optimized algorithm
        elapsed = end_time - start_time
        assert elapsed < 90, f"Tree construction took {elapsed:.2f}s, expected < 90s"
        
        # Should have events
        assert len(perf_analyzer.tree.events) > 1000000
    
    def test_kernel_launcher_detection(self, trace_file_path, skip_if_no_trace_file):
        """Test that kernel launchers are detected correctly"""
        perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(trace_file_path)
        
        kernel_launchers = perf_analyzer.get_kernel_launchers()
        
        # Should find kernel launchers
        assert len(kernel_launchers) > 1000
        
        # Each launcher should have required fields
        for launcher in kernel_launchers[:10]:  # Check first 10
            assert 'name' in launcher
            assert 'UID' in launcher
            assert 'total_direct_kernel_time' in launcher
            assert 'direct_kernel_count' in launcher
    
    def test_dataframe_generation(self, trace_file_path, skip_if_no_trace_file):
        """Test DataFrame generation"""
        perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(trace_file_path)
        
        df = perf_analyzer.get_df_kernel_launchers()
        
        # Should have rows and columns
        assert len(df) > 1000
        assert 'name' in df.columns
        assert 'total_direct_kernel_time' in df.columns
        assert 'direct_kernel_count' in df.columns


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])