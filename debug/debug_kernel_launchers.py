#!/usr/bin/env python3
"""
Debug script to check kernel launcher detection step by step
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_kernel_launchers():
    """Debug get_kernel_launchers step by step"""
    print("🔍 Debugging get_kernel_launchers() Step by Step")
    print("=" * 60)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    # Manually implement get_kernel_launchers() with debugging
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    print(f"📊 CPU ops found: {len(cpu_ops)}")
    
    kernel_launchers = []
    total_kernels_found = 0
    
    # Process first 5 CPU ops only for debugging
    for i, event in enumerate(cpu_ops[:5]):
        print(f"\n🔍 Processing CPU op #{i}: {event.get('name', 'UNKNOWN')}")
        
        kernel_launcher = False
        list_kernels = []
        
        for child_UID in event.get('children', []):
            child = perf_analyzer.tree.events_by_uid[child_UID]
            child_category = perf_analyzer.event_to_category(child)
            print(f"   Child {child_UID}: {child.get('name', 'UNKNOWN')} - {child_category}")
            
            for grand_child_UID in child.get('children', []):
                grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                grand_child_category = perf_analyzer.event_to_category(grand_child)
                
                is_kernel = grand_child_category == 'kernel'
                is_nccl = 'nccl' in grand_child['name']
                should_include = is_kernel and (True or not is_nccl)  # include_nccl=True
                
                print(f"      Grandchild {grand_child_UID}: {grand_child.get('name', 'UNKNOWN')} - {grand_child_category}")
                print(f"         is_kernel={is_kernel}, is_nccl={is_nccl}, should_include={should_include}")
                
                if should_include:
                    kernel_launcher = True
                    list_kernels.append(grand_child)
                    total_kernels_found += 1
        
        print(f"   → kernel_launcher={kernel_launcher}, kernels_found={len(list_kernels)}")
        
        if kernel_launcher:
            print(f"   → Adding to kernel_launchers list")
            kernel_launchers.append(event)
        else:
            print(f"   → NOT adding to kernel_launchers (no kernels found)")
            
    print(f"\n📊 Summary:")
    print(f"   Total kernel launchers found: {len(kernel_launchers)}")
    print(f"   Total kernels found: {total_kernels_found}")
    
    return len(kernel_launchers), total_kernels_found

if __name__ == "__main__":
    debug_kernel_launchers()