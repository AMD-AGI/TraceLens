#!/usr/bin/env python3
"""
Debug script to check kernel assignment in hash distribution
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.util import DataLoader, TraceEventUtils
from TraceLens.TreePerf.jax_analyses import JaxAnalyses

def debug_kernel_assignment():
    """Debug the kernel assignment process"""
    print("🔍 Debugging Kernel Assignment in Hash Distribution")
    print("=" * 60)
    
    # Load data manually like the tree construction does
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    data = DataLoader.load_data(xplane_path)
    trace_events = data['traceEvents']
    
    categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
    non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
    
    # Separate events by category (same as in tree construction)
    cpu_ops = []
    kernels = []
    other_events = []
    
    for event in non_metadata_events:
        category = categorizer(event)
        if category == 'cpu_op':
            cpu_ops.append(event)
        elif category in {'kernel', 'memcpy', 'memset'}:
            kernels.append(event)
        else:
            other_events.append(event)
    
    print(f"📊 Found {len(cpu_ops)} CPU ops, {len(kernels)} kernels, {len(other_events)} other events")
    
    if len(cpu_ops) == 0:
        print("❌ No CPU ops found!")
        return
    
    if len(kernels) == 0:
        print("❌ No kernels found!")
        return
    
    # Test hash distribution like in the tree construction
    cpu_op_kernels = [[] for _ in range(len(cpu_ops))]
    for kernel in kernels:
        cpu_index = hash(kernel.get('name', '')) % len(cpu_ops)
        cpu_op_kernels[cpu_index].append(kernel)
    
    # Check distribution
    total_assigned = sum(len(k_list) for k_list in cpu_op_kernels)
    print(f"📊 Total kernels assigned: {total_assigned} / {len(kernels)}")
    
    # Check first few CPU ops
    for i in range(min(10, len(cpu_ops))):
        print(f"   CPU op #{i}: '{cpu_ops[i].get('name', 'UNKNOWN')}' → {len(cpu_op_kernels[i])} kernels")
        if len(cpu_op_kernels[i]) > 0:
            print(f"      First kernel: {cpu_op_kernels[i][0].get('name', 'UNKNOWN')}")
    
    # Check empty assignments
    empty_assignments = [i for i, k_list in enumerate(cpu_op_kernels) if len(k_list) == 0]
    print(f"📊 CPU ops with 0 kernels: {len(empty_assignments)} / {len(cpu_ops)}")
    
    # Check non-empty assignments
    non_empty_assignments = [i for i, k_list in enumerate(cpu_op_kernels) if len(k_list) > 0]
    print(f"📊 CPU ops with >0 kernels: {len(non_empty_assignments)} / {len(cpu_ops)}")
    
    if len(non_empty_assignments) > 0:
        print(f"   Examples of non-empty assignments:")
        for i in non_empty_assignments[:5]:
            print(f"      CPU op #{i}: '{cpu_ops[i].get('name', 'UNKNOWN')}' → {len(cpu_op_kernels[i])} kernels")
    
    # Check kernel names and their hash values
    print(f"\n🔍 Sample kernel names and their hash distribution:")
    for i, kernel in enumerate(kernels[:10]):
        name = kernel.get('name', '')
        cpu_index = hash(name) % len(cpu_ops)
        print(f"   Kernel #{i}: '{name}' → CPU op #{cpu_index} (hash={hash(name)})")

if __name__ == "__main__":
    debug_kernel_assignment()