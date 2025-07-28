#!/usr/bin/env python3
"""
Show JAX tree structure by finding a matrix multiplication kernel and its parents
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def show_tree_structure():
    print("Loading JAX TreePerfAnalyzer...")
    
    # Create analyzer from xplane.pb file
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(
        "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    )
    
    print("Analyzing tree structure...")
    tree = perf_analyzer.tree
    
    # Find matrix multiplication kernels
    print("\n=== Searching for Matrix Multiplication Kernels ===")
    gemm_kernels = []
    
    for event in tree.events:
        if perf_analyzer.event_to_category(event) == 'kernel':
            name = event.get('name', '').lower()
            if any(keyword in name for keyword in ['gemm', 'matmul', 'dot', 'conv']):
                gemm_kernels.append(event)
                if len(gemm_kernels) >= 5:  # Limit to first 5 for display
                    break
    
    print(f"Found {len(gemm_kernels)} matrix multiplication kernels")
    for i, kernel in enumerate(gemm_kernels):
        print(f"  {i+1}. {kernel.get('name', 'UNKNOWN')} (UID: {kernel.get('UID')})")
    
    if not gemm_kernels:
        print("No GEMM kernels found. Showing any kernel instead...")
        for event in tree.events:
            if perf_analyzer.event_to_category(event) == 'kernel':
                gemm_kernels = [event]
                break
    
    if not gemm_kernels:
        print("No kernels found at all!")
        return
    
    # Choose the first GEMM kernel as our leaf
    leaf_kernel = gemm_kernels[0]
    print(f"\n=== Tree Structure for Kernel: {leaf_kernel.get('name')} ===")
    
    # Find parents of this kernel
    kernel_uid = leaf_kernel.get('UID')
    print(f"🍃 LEAF (Kernel): UID={kernel_uid}, Name='{leaf_kernel.get('name')}', Category={perf_analyzer.event_to_category(leaf_kernel)}")
    
    # Find runtime event parent (should have this kernel as child)
    runtime_parent = None
    for event in tree.events:
        if 'children' in event and kernel_uid in event.get('children', []):
            runtime_parent = event
            break
    
    if runtime_parent:
        runtime_uid = runtime_parent.get('UID')
        print(f"🌿 PARENT (Runtime): UID={runtime_uid}, Name='{runtime_parent.get('name')}', Category={perf_analyzer.event_to_category(runtime_parent)}")
        
        # Find CPU op grandparent (should have runtime event as child)
        cpu_grandparent = None
        for event in tree.events:
            if 'children' in event and runtime_uid in event.get('children', []):
                cpu_grandparent = event
                break
        
        if cpu_grandparent:
            cpu_uid = cpu_grandparent.get('UID')
            print(f"🌳 GRANDPARENT (CPU Op): UID={cpu_uid}, Name='{cpu_grandparent.get('name')}', Category={perf_analyzer.event_to_category(cpu_grandparent)}")
            
            # Show the complete hierarchy
            print(f"\n=== Complete Hierarchy ===")
            print(f"CPU Op '{cpu_grandparent.get('name')}' (UID {cpu_uid})")
            print(f"  └─ Runtime '{runtime_parent.get('name')}' (UID {runtime_uid})")
            print(f"      └─ Kernel '{leaf_kernel.get('name')}' (UID {kernel_uid})")
            
            # Show additional details
            print(f"\n=== Event Details ===")
            
            print(f"CPU Op Event:")
            print(f"  - Name: {cpu_grandparent.get('name')}")
            print(f"  - Category: {perf_analyzer.event_to_category(cpu_grandparent)}")
            print(f"  - Children: {cpu_grandparent.get('children', [])}")
            print(f"  - Args keys: {list(cpu_grandparent.get('args', {}).keys())}")
            
            print(f"\nRuntime Event:")
            print(f"  - Name: {runtime_parent.get('name')}")
            print(f"  - Category: {perf_analyzer.event_to_category(runtime_parent)}")
            print(f"  - Children: {runtime_parent.get('children', [])}")
            print(f"  - Duration: {runtime_parent.get('dur', 'N/A')}")
            
            print(f"\nKernel Event:")
            print(f"  - Name: {leaf_kernel.get('name')}")
            print(f"  - Category: {perf_analyzer.event_to_category(leaf_kernel)}")
            print(f"  - Duration: {leaf_kernel.get('dur', 'N/A')}")
            print(f"  - Args keys: {list(leaf_kernel.get('args', {}).keys())}")
            
        else:
            print("❌ No CPU op grandparent found for this runtime event")
    else:
        print("❌ No runtime parent found for this kernel")
    
    print(f"\n=== Tree Statistics ===")
    print(f"Total events: {len(tree.events)}")
    
    # Count by category
    categories = {}
    for event in tree.events:
        cat = perf_analyzer.event_to_category(event)
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    show_tree_structure()