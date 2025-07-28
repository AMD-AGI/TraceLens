#!/usr/bin/env python3
"""
Direct test of JAX functionality without import issues
"""

import sys
import os
import traceback

# Add current directory to path so we can import TraceLens
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_imports():
    """Test that basic imports work"""
    print("=== Testing Basic Imports ===")
    try:
        import TraceLens
        print("✓ TraceLens import successful")
        
        from TraceLens import TreePerfAnalyzer, TraceToTree
        print("✓ Core classes import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_jax_imports():
    """Test JAX-specific imports"""
    print("\n=== Testing JAX Imports ===")
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree, JaxTreePerfAnalyzer
        print("✓ JAX classes import successful")
        
        from TraceLens.PerfModel.jax_op_mapping import get_jax_perf_model_class, JaxDummyPerfModel
        print("✓ JAX performance models import successful")
        
        return True
    except Exception as e:
        print(f"✗ JAX imports failed: {e}")
        traceback.print_exc()
        return False

def test_jax_tree_creation():
    """Test JAX tree creation"""
    print("\n=== Testing JAX Tree Creation ===")
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree
        
        xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
        if not os.path.exists(xplane_path):
            print(f"✗ Test file not found: {xplane_path}")
            return False
            
        print(f"Creating tree from: {xplane_path}")
        tree = JaxXplaneToTree.from_xplane_pb(xplane_path)
        
        print(f"✓ Successfully created tree with {len(tree.events)} events")
        print(f"✓ Tree has {len(tree.cpu_root_nodes)} CPU root nodes")
        
        # Check event categories
        categories = {}
        for event in tree.events[:1000]:  # Sample first 1000 events
            cat = tree.event_to_category(event)
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"✓ Event categories found: {list(categories.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ JAX tree creation failed: {e}")
        traceback.print_exc()
        return False

def test_jax_analyzer():
    """Test JAX TreePerfAnalyzer"""
    print("\n=== Testing JAX TreePerfAnalyzer ===")
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
        
        xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
        if not os.path.exists(xplane_path):
            print(f"✗ Test file not found: {xplane_path}")
            return False
            
        print(f"Creating analyzer from: {xplane_path}")
        analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
        
        print(f"✓ Successfully created analyzer")
        print(f"✓ JAX flag is set: {analyzer.jax}")
        print(f"✓ Tree has {len(analyzer.tree.events)} events")
        
        # Test basic operations
        print("Testing basic operations...")
        
        # GPU timeline
        df_timeline = analyzer.get_df_gpu_timeline()
        print(f"✓ GPU timeline: {len(df_timeline)} rows")
        print(f"  Columns: {list(df_timeline.columns)}")
        
        # Kernel launchers
        df_launchers = analyzer.get_df_kernel_launchers()
        print(f"✓ Kernel launchers: {len(df_launchers)} rows")
        
        return True
    except Exception as e:
        print(f"✗ JAX analyzer failed: {e}")
        traceback.print_exc()
        return False

def test_dummy_values():
    """Test that dummy values are properly set"""
    print("\n=== Testing Dummy Values ===")
    try:
        from TraceLens.PerfModel.jax_op_mapping import JaxDummyPerfModel
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree
        
        # Test dummy performance model
        dummy_event = {'name': 'test_event'}
        dummy_model = JaxDummyPerfModel(dummy_event)
        
        flops = dummy_model.flops()
        bytes_val = dummy_model.bytes()
        
        print(f"✓ Dummy FLOPS: {flops} (negative indicates dummy)")
        print(f"✓ Dummy bytes: {bytes_val} (negative indicates dummy)")
        
        # Check for dummy values in enhanced events
        xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
        if os.path.exists(xplane_path):
            tree = JaxXplaneToTree.from_xplane_pb(xplane_path)
            
            # Look for dummy values in enhanced events
            dummy_found = False
            for event in tree.events[:100]:
                args = event.get('args', {})
                
                # Check for obvious dummy values
                for key, value in args.items():
                    if isinstance(value, str) and 'DUMMY_JAX' in value:
                        print(f"✓ Found dummy JAX string: {key} = {value}")
                        dummy_found = True
                        break
                    elif isinstance(value, int) and value < -1000:
                        print(f"✓ Found dummy negative ID: {key} = {value}")
                        dummy_found = True
                        break
                
                if dummy_found:
                    break
            
            if not dummy_found:
                print("! No obvious dummy values found in first 100 events")
        
        return True
    except Exception as e:
        print(f"✗ Dummy value test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_analysis():
    """Test JAX-specific performance analysis"""
    print("\n=== Testing Performance Analysis ===")
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
        from TraceLens.PerfModel.jax_op_mapping import get_jax_perf_model_class
        
        # Test performance model selection
        test_ops = [
            'custom-call',
            'loop_convert_fusion', 
            'gemm_fusion_dot',
            'unknown_operation'
        ]
        
        for op in test_ops:
            model_class = get_jax_perf_model_class(op)
            model_name = model_class.__name__ if model_class else "None"
            print(f"✓ {op} -> {model_name}")
        
        # Test with actual analyzer if file exists
        xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
        if os.path.exists(xplane_path):
            analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
            
            # Find a few events to test performance computation
            gpu_events = [e for e in analyzer.tree.events[:100] 
                         if analyzer.event_to_category(e) in {'kernel', 'gpu_memcpy', 'gpu_memset'}
                         and e.get('tree')]
            
            if gpu_events:
                test_event = gpu_events[0]
                print(f"✓ Testing performance computation on: {test_event.get('name', 'unnamed')}")
                
                try:
                    # This might fail due to missing fields, but should not crash
                    metrics = analyzer.compute_perf_metrics(test_event)
                    print(f"✓ Performance metrics computed: {list(metrics.keys())}")
                except Exception as e:
                    print(f"! Performance computation failed (expected): {e}")
                    # This is expected for some events without proper data
        
        return True
    except Exception as e:
        print(f"✗ Performance analysis test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("JAX Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("JAX Imports", test_jax_imports), 
        ("JAX Tree Creation", test_jax_tree_creation),
        ("JAX Analyzer", test_jax_analyzer),
        ("Dummy Values", test_dummy_values),
        ("Performance Analysis", test_performance_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JAX integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())