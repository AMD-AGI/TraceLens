#!/usr/bin/env python3
"""
Debug script to check event field formats (ts, dur, t_end)
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_event_fields():
    """Debug event field formats"""
    print("🔍 Debugging Event Field Formats (ts, dur, t_end)")
    print("=" * 55)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    # Find some kernel events and check their fields
    kernel_events = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'kernel']
    print(f"📊 Kernel events found: {len(kernel_events)}")
    
    if len(kernel_events) > 0:
        print(f"\n🔍 First 5 kernel events:")
        for i, event in enumerate(kernel_events[:5]):
            print(f"   Kernel #{i}: {event.get('name', 'UNKNOWN')}")
            print(f"      UID: {event.get('UID', 'NO_UID')}")
            print(f"      Keys: {list(event.keys())}")
            print(f"      ts: {event.get('ts', 'MISSING')}")
            print(f"      dur: {event.get('dur', 'MISSING')}")
            print(f"      t_end: {event.get('t_end', 'MISSING')}")
            
            # Check if we can compute t_end from ts + dur
            if 'ts' in event and 'dur' in event:
                computed_t_end = event['ts'] + event['dur']
                print(f"      computed t_end: {computed_t_end}")
    
    # Test the GPU event analysis issue directly
    print(f"\n🔍 Testing GPUEventAnalyser fields directly...")
    
    # Find CPU ops with children
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    cpu_ops_with_children = [e for e in cpu_ops if len(e.get('children', [])) > 0]
    
    print(f"📊 CPU ops with children: {len(cpu_ops_with_children)} / {len(cpu_ops)}")
    
    if len(cpu_ops_with_children) > 0:
        test_cpu_op = cpu_ops_with_children[0]
        print(f"   Testing CPU op: {test_cpu_op.get('name', 'UNKNOWN')}")
        print(f"   Children: {len(test_cpu_op.get('children', []))}")
        
        # Manually extract kernels like get_kernel_launchers does
        list_kernels = []
        for child_UID in test_cpu_op.get('children', []):
            if child_UID in perf_analyzer.tree.events_by_uid:
                child = perf_analyzer.tree.events_by_uid[child_UID]
                for grand_child_UID in child.get('children', []):
                    if grand_child_UID in perf_analyzer.tree.events_by_uid:
                        grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                        if perf_analyzer.event_to_category(grand_child) == 'kernel':
                            list_kernels.append(grand_child)
        
        print(f"   Kernels found: {len(list_kernels)}")
        
        if len(list_kernels) > 0:
            print(f"   First kernel fields:")
            first_kernel = list_kernels[0]
            print(f"      Name: {first_kernel.get('name', 'UNKNOWN')}")
            print(f"      ts: {first_kernel.get('ts', 'MISSING')}")
            print(f"      dur: {first_kernel.get('dur', 'MISSING')}")
            print(f"      t_end: {first_kernel.get('t_end', 'MISSING')}")
            
            # This is what GPUEventAnalyser checks
            missing_ts = 'ts' not in first_kernel
            missing_t_end = 't_end' not in first_kernel
            print(f"      Missing ts: {missing_ts}")
            print(f"      Missing t_end: {missing_t_end}")
            
            if missing_t_end and 'ts' in first_kernel and 'dur' in first_kernel:
                print(f"      Should add t_end = ts + dur = {first_kernel['ts'] + first_kernel['dur']}")

if __name__ == "__main__":
    debug_event_fields()