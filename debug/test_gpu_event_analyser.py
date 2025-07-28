#!/usr/bin/env python3
"""
Test GPUEventAnalyser directly with JAX kernel events
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser

def test_gpu_event_analyser():
    """Test GPUEventAnalyser directly with JAX kernel events"""
    print("🧪 Testing GPUEventAnalyser with JAX kernel events")
    print("=" * 50)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    # Get CPU ops and find one with kernels
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    cpu_ops_with_children = [e for e in cpu_ops if len(e.get('children', [])) > 0]
    
    if len(cpu_ops_with_children) == 0:
        print("❌ No CPU ops with children found")
        return
    
    # Find a CPU op that actually has kernels (not just runtime events with no children)
    test_cpu_op = None
    for cpu_op in cpu_ops_with_children:
        kernel_count = 0
        for child_UID in cpu_op.get('children', []):
            if child_UID in perf_analyzer.tree.events_by_uid:
                child = perf_analyzer.tree.events_by_uid[child_UID]
                for grand_child_UID in child.get('children', []):
                    if grand_child_UID in perf_analyzer.tree.events_by_uid:
                        grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                        if perf_analyzer.event_to_category(grand_child) == 'kernel':
                            kernel_count += 1
        if kernel_count > 0:
            test_cpu_op = cpu_op
            print(f"📊 Testing CPU op: {test_cpu_op.get('name', 'UNKNOWN')} (has {kernel_count} kernels)")
            break
    
    if test_cpu_op is None:
        print("❌ No CPU ops with actual kernels found")
        return
    
    # Extract kernels manually like get_kernel_launchers does
    list_kernels = []
    for child_UID in test_cpu_op.get('children', []):
        if child_UID in perf_analyzer.tree.events_by_uid:
            child = perf_analyzer.tree.events_by_uid[child_UID]
            for grand_child_UID in child.get('children', []):
                if grand_child_UID in perf_analyzer.tree.events_by_uid:
                    grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                    if perf_analyzer.event_to_category(grand_child) == 'kernel':
                        list_kernels.append(grand_child)
    
    print(f"📊 Kernels extracted: {len(list_kernels)}")
    
    if len(list_kernels) == 0:
        print("❌ No kernels found")
        return
    
    # Test GPUEventAnalyser initialization
    print(f"\n🔍 Testing GPUEventAnalyser...")
    
    try:
        gpu_analyser = GPUEventAnalyser(list_kernels)
        print(f"✅ GPUEventAnalyser created successfully")
        
        # Test get_gpu_event_lists()
        gpu_event_lists = gpu_analyser.get_gpu_event_lists()
        print(f"✅ get_gpu_event_lists() completed")
        print(f"   Keys: {list(gpu_event_lists.keys())}")
        
        for key, events in gpu_event_lists.items():
            print(f"   {key}: {len(events)} events")
        
        # Test the specific check that's failing
        if 'all_gpu' in gpu_event_lists:
            all_gpu_count = len(gpu_event_lists['all_gpu'])
            print(f"   all_gpu count: {all_gpu_count}")
            
            if all_gpu_count == 0:
                print("❌ This is the problem - all_gpu is empty!")
                
                # Check what categories our kernels fall into
                print(f"   Debugging kernel categorization...")
                for i, kernel in enumerate(list_kernels[:3]):
                    print(f"      Kernel #{i}: {kernel.get('name', 'UNKNOWN')}")
                    print(f"         pid: {kernel.get('pid', 'MISSING')}")
                    print(f"         tid: {kernel.get('tid', 'MISSING')}")
                    print(f"         Has ts: {'ts' in kernel}")
                    print(f"         Has t_end: {'t_end' in kernel}")
                    
            else:
                print("✅ all_gpu has events, continuing...")
                
                # Test compute_metrics()
                try:
                    metrics = gpu_analyser.compute_metrics()
                    print(f"✅ compute_metrics() completed successfully!")
                    print(f"   Metrics: {list(metrics.keys())}")
                    if 'busy_time' in metrics:
                        print(f"   busy_time: {metrics['busy_time']}")
                        
                except Exception as e:
                    print(f"❌ compute_metrics() failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("❌ No 'all_gpu' key in results")
            
    except Exception as e:
        print(f"❌ GPUEventAnalyser failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpu_event_analyser()