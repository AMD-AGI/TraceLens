#!/usr/bin/env python3
"""
Debug script to check kernel detection in optimized JAX tree
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_kernel_detection():
    """Debug kernel detection in JAX tree"""
    print("🔍 Debugging Kernel Detection in Optimized JAX Tree")
    print("=" * 60)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    print(f"📊 Events by UID: {len(perf_analyzer.tree.events_by_uid):,}")
    
    # Check CPU ops
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    print(f"📊 CPU ops found: {len(cpu_ops)}")
    
    if len(cpu_ops) > 0:
        first_cpu_op = cpu_ops[0]
        print(f"🔍 First CPU op: {first_cpu_op.get('name', 'UNKNOWN')}")
        print(f"   UID: {first_cpu_op.get('UID', 'NO_UID')}")
        print(f"   Children: {first_cpu_op.get('children', [])}")
        
        # Check children
        children_uids = first_cpu_op.get('children', [])
        print(f"   Children count: {len(children_uids)}")
        
        if len(children_uids) > 0:
            first_child_uid = children_uids[0]
            if first_child_uid in perf_analyzer.tree.events_by_uid:
                first_child = perf_analyzer.tree.events_by_uid[first_child_uid]
                print(f"   First child: {first_child.get('name', 'UNKNOWN')}")
                print(f"   Child category: {perf_analyzer.event_to_category(first_child)}")
                print(f"   Child children: {first_child.get('children', [])}")
                
                # Check grandchildren (should be kernels)
                grandchildren_uids = first_child.get('children', [])
                print(f"   Grandchildren count: {len(grandchildren_uids)}")
                
                if len(grandchildren_uids) > 0:
                    first_grandchild_uid = grandchildren_uids[0]
                    if first_grandchild_uid in perf_analyzer.tree.events_by_uid:
                        first_grandchild = perf_analyzer.tree.events_by_uid[first_grandchild_uid]
                        print(f"   First grandchild: {first_grandchild.get('name', 'UNKNOWN')}")
                        print(f"   Grandchild category: {perf_analyzer.event_to_category(first_grandchild)}")
                        print(f"   Has ts: {'ts' in first_grandchild}")
                        print(f"   Has t_end: {'t_end' in first_grandchild}")
                    else:
                        print(f"   ⚠️  Grandchild UID {first_grandchild_uid} not found in events_by_uid!")
                else:
                    print("   ⚠️  No grandchildren found!")
            else:
                print(f"   ⚠️  Child UID {first_child_uid} not found in events_by_uid!")
        else:
            print("   ⚠️  No children found!")
    
    # Try to manually run kernel launcher detection
    print("\n🚀 Testing get_kernel_launchers()...")
    try:
        kernel_launchers = perf_analyzer.get_kernel_launchers()
        print(f"✅ Found {len(kernel_launchers)} kernel launchers")
        if len(kernel_launchers) > 0:
            first_launcher = kernel_launchers[0]
            print(f"   First launcher: {first_launcher.get('name', 'UNKNOWN')}")
            print(f"   Direct kernel count: {first_launcher.get('direct_kernel_count', 0)}")
    except Exception as e:
        print(f"❌ Error in get_kernel_launchers(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_kernel_detection()