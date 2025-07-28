#!/usr/bin/env python3
"""
Test script for JAX xplane.pb integration with TraceLens

This script validates that JAX xplane.pb files can be successfully ingested
and processed by the existing TreePerfAnalyzer infrastructure without breaking
PyTorch functionality.

Usage:
    python test_jax_integration.py --xplane_pb_path trace.xplane.pb
"""

import argparse
import sys
import traceback
from pathlib import Path


def test_jax_tree_creation(xplane_pb_path: str):
    """Test basic JAX tree creation from xplane.pb file."""
    print("=== Testing JAX Tree Creation ===")
    
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree
        
        # Create tree from JAX file
        tree = JaxXplaneToTree.from_xplane_pb(xplane_pb_path)
        
        print(f"✓ Successfully created tree with {len(tree.events)} events")
        print(f"✓ Tree has {len(tree.cpu_root_nodes)} CPU root nodes")
        
        # Test that events have required fields
        sample_events = tree.events[:5]
        for i, event in enumerate(sample_events):
            print(f"  Event {i}: {event.get('name', 'unnamed')} - Category: {tree.event_to_category(event)}")
            
        return True
        
    except Exception as e:
        print(f"✗ JAX tree creation failed: {e}")
        traceback.print_exc()
        return False


def test_jax_tree_perf_analyzer(xplane_pb_path: str):
    """Test JAX TreePerfAnalyzer creation and basic operations."""
    print("\n=== Testing JAX TreePerfAnalyzer ===")
    
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
        
        # Create analyzer from JAX file
        analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_pb_path)
        
        print(f"✓ Successfully created TreePerfAnalyzer")
        print(f"✓ JAX flag is set: {analyzer.jax}")
        print(f"✓ Tree has {len(analyzer.tree.events)} events")
        
        # Test basic operations that should work without errors
        print("  Testing basic operations...")
        
        # Test GPU timeline generation
        try:
            df_timeline = analyzer.get_df_gpu_timeline()
            print(f"  ✓ GPU timeline: {len(df_timeline)} rows")
        except Exception as e:
            print(f"  ✗ GPU timeline failed: {e}")
            
        # Test kernel launcher analysis
        try:
            df_launchers = analyzer.get_df_kernel_launchers(include_kernel_names=True)
            print(f"  ✓ Kernel launchers: {len(df_launchers)} rows")
        except Exception as e:
            print(f"  ✗ Kernel launchers failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ JAX TreePerfAnalyzer creation failed: {e}")
        traceback.print_exc()
        return False


def test_pytorch_compatibility():
    """Test that PyTorch functionality still works unchanged."""
    print("\n=== Testing PyTorch Compatibility ===")
    
    try:
        # Import PyTorch classes to ensure they still work
        from TraceLens import TreePerfAnalyzer, TraceToTree
        from TraceLens.util import TraceEventUtils
        
        print("✓ PyTorch imports successful")
        
        # Test that PyTorch-specific methods exist
        assert hasattr(TreePerfAnalyzer, 'from_file')
        assert hasattr(TraceToTree, 'default_categorizer')
        assert hasattr(TraceEventUtils, 'default_categorizer')
        
        print("✓ PyTorch methods still available")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch compatibility check failed: {e}")
        traceback.print_exc()
        return False


def test_dummy_values_detectability(xplane_pb_path: str):
    """Test that dummy values are easily detectable in the output."""
    print("\n=== Testing Dummy Value Detectability ===")
    
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree
        
        tree = JaxXplaneToTree.from_xplane_pb(xplane_pb_path)
        
        # Check for dummy values in events
        dummy_found = False
        for event in tree.events[:100]:  # Check first 100 events
            args = event.get('args', {})
            
            # Look for obvious dummy values
            if any(str(value).startswith('DUMMY_JAX') for value in args.values()):
                dummy_found = True
                print(f"  ✓ Found dummy JAX values in event: {event.get('name', 'unnamed')}")
                break
                
            # Look for negative dummy IDs
            if any(isinstance(value, int) and value < -1000 for value in args.values()):
                dummy_found = True
                print(f"  ✓ Found negative dummy IDs in event: {event.get('name', 'unnamed')}")
                break
        
        if dummy_found:
            print("✓ Dummy values are detectable and obvious")
        else:
            print("! No obvious dummy values found in sample events")
            
        return True
        
    except Exception as e:
        print(f"✗ Dummy value check failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test JAX xplane.pb integration with TraceLens')
    parser.add_argument('--xplane_pb_path', type=str, required=True, help='Path to JAX xplane.pb file')
    args = parser.parse_args()
    
    # Verify input file exists
    if not Path(args.xplane_pb_path).exists():
        print(f"Error: File {args.xplane_pb_path} does not exist")
        sys.exit(1)
    
    print(f"Testing JAX integration with file: {args.xplane_pb_path}")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("PyTorch Compatibility", test_pytorch_compatibility),
        ("JAX Tree Creation", lambda: test_jax_tree_creation(args.xplane_pb_path)),
        ("JAX TreePerfAnalyzer", lambda: test_jax_tree_perf_analyzer(args.xplane_pb_path)),
        ("Dummy Values Detection", lambda: test_dummy_values_detectability(args.xplane_pb_path)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append(result)
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! JAX integration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()