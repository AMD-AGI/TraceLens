#!/usr/bin/env python3
"""
Debug script to check CPU op #31 which has kernels
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def debug_cpu_op_31():
    """Debug CPU op #31 which should have kernels"""
    print("🔍 Debugging CPU Op #31 (has 464 kernels)")
    print("=" * 50)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    # Find CPU ops and check CPU op #31 specifically
    cpu_ops = [e for e in perf_analyzer.tree.events if perf_analyzer.event_to_category(e) == 'cpu_op']
    print(f"📊 CPU ops found: {len(cpu_ops)}")
    
    if len(cpu_ops) <= 31:
        print("❌ Not enough CPU ops found")
        return
    
    # Check CPU op #31
    cpu_op_31 = cpu_ops[31]
    print(f"\n🔍 CPU op #31: {cpu_op_31.get('name', 'UNKNOWN')}")
    print(f"   UID: {cpu_op_31.get('UID', 'NO_UID')}")
    print(f"   Children: {len(cpu_op_31.get('children', []))}")
    
    kernel_count = 0
    for child_UID in cpu_op_31.get('children', []):
        if child_UID in perf_analyzer.tree.events_by_uid:
            child = perf_analyzer.tree.events_by_uid[child_UID]
            child_category = perf_analyzer.event_to_category(child)
            
            for grand_child_UID in child.get('children', []):
                if grand_child_UID in perf_analyzer.tree.events_by_uid:
                    grand_child = perf_analyzer.tree.events_by_uid[grand_child_UID]
                    grand_child_category = perf_analyzer.event_to_category(grand_child)
                    
                    if grand_child_category == 'kernel':
                        kernel_count += 1
                        if kernel_count <= 5:  # Show first 5 kernels
                            print(f"   Kernel #{kernel_count}: {grand_child.get('name', 'UNKNOWN')}")
    
    print(f"\n📊 Total kernels found for CPU op #31: {kernel_count}")
    
    # Now test get_kernel_launchers with more targeted debugging
    print("\n🔍 Testing get_kernel_launchers() (should find CPU op #31)...")
    try:
        kernel_launchers = perf_analyzer.get_kernel_launchers()
        print(f"✅ SUCCESS: Found {len(kernel_launchers)} kernel launchers")
        
        # Check if CPU op #31 is in the results
        cpu_op_31_uid = cpu_op_31.get('UID', -1)
        found_cpu_op_31 = any(kl.get('UID') == cpu_op_31_uid for kl in kernel_launchers)
        print(f"   CPU op #31 found in kernel launchers: {found_cpu_op_31}")
        
        if found_cpu_op_31:
            print("✅ CPU op #31 correctly detected as kernel launcher!")
        else:
            print("❌ CPU op #31 NOT found in kernel launchers (unexpected)")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    debug_cpu_op_31()