#!/usr/bin/env python3
"""
Debug script to check the 'cat' field in JAX kernel events
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_cat_field():
    """Debug the 'cat' field in JAX kernel events"""
    print("🔍 Debugging 'cat' field in JAX kernel events")
    print("=" * 50)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    # Find kernel events
    kernel_events = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'kernel']
    print(f"📊 Kernel events found: {len(kernel_events)}")
    
    if len(kernel_events) > 0:
        print(f"\n🔍 First 5 kernel events - checking 'cat' field:")
        for i, event in enumerate(kernel_events[:5]):
            print(f"   Kernel #{i}: {event.get('name', 'UNKNOWN')}")
            cat_value = event.get('cat', 'MISSING')
            print(f"      cat: '{cat_value}'")
            print(f"      categorizer says: '{perf_analyzer.event_to_category(event)}'")
            
            # Show all fields for debugging
            print(f"      all keys: {list(event.keys())}")
            
            # Check what GPUEventAnalyser expects
            expected_cats = {'kernel', 'gpu_memcpy', 'gpu_memset'}
            cat_match = cat_value in expected_cats
            print(f"      cat matches GPUEventAnalyser expectations: {cat_match}")
    
    # Also check synthetic runtime events
    runtime_events = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cuda_runtime']
    print(f"\n📊 Runtime events found: {len(runtime_events)}")
    
    if len(runtime_events) > 0:
        print(f"\n🔍 First 3 runtime events - checking 'cat' field:")
        for i, event in enumerate(runtime_events[:3]):
            print(f"   Runtime #{i}: {event.get('name', 'UNKNOWN')}")
            cat_value = event.get('cat', 'MISSING')
            print(f"      cat: '{cat_value}'")
            print(f"      categorizer says: '{perf_analyzer.event_to_category(event)}'")

if __name__ == "__main__":
    debug_cat_field()