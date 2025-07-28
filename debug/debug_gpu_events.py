#!/usr/bin/env python3
"""
Debug script to check GPU events retrieval in kernel launcher detection
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_gpu_events():
    """Debug GPU events in kernel launcher detection"""
    print("🔍 Debugging GPU Events in Kernel Launcher Detection")
    print("=" * 70)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    # Find CPU ops and check their children structure
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    print(f"📊 CPU ops found: {len(cpu_ops)}")
    
    total_kernels_found = 0
    
    # Manually recreate the kernel launcher detection logic
    for i, cpu_op in enumerate(cpu_ops[:5]):  # Check first 5 CPU ops only
        print(f"\n🔍 CPU op #{i}: {cpu_op.get('name', 'UNKNOWN')}")
        print(f"   UID: {cpu_op.get('UID', 'NO_UID')}")
        print(f"   Children: {len(cpu_op.get('children', []))}")
        
        list_kernels = []
        for child_UID in cpu_op.get('children', []):
            if child_UID in perf_analyzer.tree.events_by_uid:
                child = perf_analyzer.tree.events_by_uid[child_UID]
                print(f"     Child {child_UID}: {child.get('name', 'UNKNOWN')} - {perf_analyzer.event_to_category(child)}")
                print(f"       Child children: {len(child.get('children', []))}")
                
                for grand_child_UID in child.get('children', []):
                    if grand_child_UID in perf_analyzer.tree.events_by_uid:
                        grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                        category = perf_analyzer.event_to_category(grand_child)
                        print(f"         Grandchild {grand_child_UID}: {grand_child.get('name', 'UNKNOWN')} - {category}")
                        
                        # Check what GPUEventAnalyser expects
                        print(f"           Has 'ts': {'ts' in grand_child}")
                        print(f"           Has 't_end': {'t_end' in grand_child}")
                        print(f"           Has 'dur': {'dur' in grand_child}")
                        
                        if category == 'kernel':
                            list_kernels.append(grand_child)
                            total_kernels_found += 1
                    else:
                        print(f"         ⚠️  Grandchild UID {grand_child_UID} not found in events_by_uid!")
            else:
                print(f"     ⚠️  Child UID {child_UID} not found in events_by_uid!")
        
        print(f"   Kernels found for this CPU op: {len(list_kernels)}")
        
        if len(list_kernels) > 0:
            print(f"   First kernel example:")
            first_kernel = list_kernels[0]
            print(f"     Name: {first_kernel.get('name', 'UNKNOWN')}")
            print(f"     Keys: {list(first_kernel.keys())}")
            if 'ts' in first_kernel and 'dur' in first_kernel:
                print(f"     ts: {first_kernel['ts']}, dur: {first_kernel['dur']}")
                print(f"     Calculated t_end: {first_kernel['ts'] + first_kernel['dur']}")
    
    print(f"\n📊 Summary:")
    print(f"   Total kernels found via tree traversal: {total_kernels_found}")
    
    return total_kernels_found

if __name__ == "__main__":
    debug_gpu_events()