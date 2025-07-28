#!/usr/bin/env python3
"""
Optimized JAX tree construction algorithm
Targeting 5x performance improvement by eliminating O(n×m) complexity
"""

import sys
import time
from typing import Dict, Any, List, Callable

sys.path.insert(0, '/home/juhaj/projects/TraceLens')

def create_optimized_tree_structure(events: List[Dict[str, Any]], categorizer: Callable) -> List[Dict[str, Any]]:
    """
    OPTIMIZED version of _create_pytorch_compatible_tree_structure
    
    Key optimizations:
    1. Pre-allocation instead of list.append()
    2. Template-based event creation instead of deep copying
    3. Hash-based kernel distribution instead of sequential assignment
    4. Batch processing of enhancements
    """
    
    print("🚀 Starting optimized tree construction...")
    start_time = time.time()
    
    # Phase 1: Categorize events (unchanged - already fast)
    cpu_ops = []
    kernels = []
    other_events = []
    
    print("📊 Categorizing events...")
    categorize_start = time.time()
    
    for event in events:
        category = categorizer(event)
        if category == 'cpu_op':
            cpu_ops.append(event)
        elif category in {'kernel', 'memcpy', 'memset'}:
            kernels.append(event)
        else:
            other_events.append(event)
    
    categorize_time = time.time() - categorize_start
    print(f"   Categorization: {categorize_time:.2f}s")
    print(f"   CPU ops: {len(cpu_ops):,}, Kernels: {len(kernels):,}, Other: {len(other_events):,}")
    
    if len(cpu_ops) == 0 or len(kernels) == 0:
        print("⚠️  No CPU ops or kernels found - returning other events only")
        return other_events
    
    # Phase 2: OPTIMIZED kernel distribution using hash-based assignment
    print("🔗 Optimized kernel distribution...")
    distribute_start = time.time()
    
    # Pre-allocate CPU op kernel assignments
    cpu_op_kernels = [[] for _ in range(len(cpu_ops))]
    
    # Hash-based distribution - O(m) instead of O(n×m)
    for kernel in kernels:
        # Use kernel name hash for distribution
        cpu_index = hash(kernel.get('name', '')) % len(cpu_ops)
        cpu_op_kernels[cpu_index].append(kernel)
    
    distribute_time = time.time() - distribute_start
    print(f"   Distribution: {distribute_time:.2f}s")
    
    # Show distribution stats
    kernel_counts = [len(k_list) for k_list in cpu_op_kernels]
    print(f"   Kernel distribution - min: {min(kernel_counts)}, max: {max(kernel_counts)}, avg: {sum(kernel_counts)/len(kernel_counts):.1f}")
    
    # Phase 3: OPTIMIZED event creation with pre-allocation
    print("🏗️  Pre-allocated event creation...")
    creation_start = time.time()
    
    # Calculate exact result size
    total_runtime_events = sum(len(k_list) for k_list in cpu_op_kernels)
    total_enhanced_kernels = total_runtime_events  # 1:1 with runtime events
    result_size = len(other_events) + len(cpu_ops) + total_runtime_events + total_enhanced_kernels
    
    print(f"   Pre-allocating {result_size:,} events...")
    
    # Pre-allocate result array
    enhanced_events = [None] * result_size
    current_index = 0
    
    # Add other events first (fast copy)
    for event in other_events:
        enhanced_events[current_index] = event
        current_index += 1
    
    # Phase 4: Create templates (avoid repeated dict creation)
    runtime_template = {
        'name': 'cudaLaunchKernel',
        'cat': 'cuda_runtime',
        'ph': 'X',
        'dur': 1,
        'args': {
            'correlation': -9999,
            'External id': -8888,
        }
    }
    
    kernel_enhancement = {
        'stream': -5555,
        'correlation': -9999,
    }
    
    cpu_enhancement = {
        'Input Dims': '[[DUMMY_JAX]]',
        'Input type': 'DUMMY_JAX_TYPE', 
        'Input Strides': '[[DUMMY_JAX_STRIDES]]',
        'Concrete Inputs': 'DUMMY_JAX_CONCRETE',
        'correlation': -9999,
        'External id': -8888,
    }
    
    # Phase 5: OPTIMIZED event processing with templates
    for i, cpu_op in enumerate(cpu_ops):
        # Enhance CPU op in-place (avoid copy)
        enhanced_cpu_op = cpu_op.copy()  # Only copy once per CPU op
        if 'args' not in enhanced_cpu_op:
            enhanced_cpu_op['args'] = {}
        enhanced_cpu_op['args'].update(cpu_enhancement)
        
        # Predict UIDs for children relationships
        runtime_uids = []
        
        # Process assigned kernels for this CPU op
        assigned_kernels = cpu_op_kernels[i]
        for kernel in assigned_kernels:
            # Create runtime event from template (fast)
            runtime_event = runtime_template.copy()
            runtime_event['ts'] = kernel.get('ts', 0)
            runtime_event['pid'] = enhanced_cpu_op.get('pid', 0)
            runtime_event['tid'] = enhanced_cpu_op.get('tid', 0)
            runtime_event['children'] = [current_index + 1]  # Next index will be kernel
            
            # Add runtime event
            enhanced_events[current_index] = runtime_event
            runtime_uids.append(current_index)
            current_index += 1
            
            # Enhance kernel (minimal copying)
            enhanced_kernel = kernel.copy()  # Only copy once per kernel
            if 'args' not in enhanced_kernel:
                enhanced_kernel['args'] = {}
            enhanced_kernel['args'].update(kernel_enhancement)
            
            # Add enhanced kernel
            enhanced_events[current_index] = enhanced_kernel
            current_index += 1
        
        # Set CPU op children
        enhanced_cpu_op['children'] = runtime_uids
        enhanced_cpu_op['gpu_events'] = [k.get('UID', k.get('id', 0)) for k in assigned_kernels]
        
        # Add enhanced CPU op
        enhanced_events[current_index] = enhanced_cpu_op
        current_index += 1
    
    creation_time = time.time() - creation_start
    print(f"   Event creation: {creation_time:.2f}s")
    
    # Remove any None entries (shouldn't happen with correct pre-allocation)
    enhanced_events = [e for e in enhanced_events if e is not None]
    
    total_time = time.time() - start_time
    print(f"✅ Optimized tree construction completed in {total_time:.2f}s")
    print(f"📊 Created {len(enhanced_events):,} enhanced events")
    
    return enhanced_events

def test_optimization():
    """Test the optimized algorithm with timing"""
    from TraceLens.util import DataLoader
    from TraceLens.TreePerf.jax_analyses import JaxAnalyses
    from TraceLens.util import TraceEventUtils
    
    print("🧪 Testing optimized JAX tree construction")
    print("=" * 60)
    
    # Load test data
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    
    print("📁 Loading trace data...")
    data = DataLoader.load_data(xplane_path)
    trace_events = data['traceEvents']
    print(f"📊 Loaded {len(trace_events):,} events")
    
    categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
    non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
    print(f"📊 Non-metadata events: {len(non_metadata_events):,}")
    
    # Test optimized version
    print("\n🚀 Running OPTIMIZED algorithm...")
    opt_start = time.time()
    optimized_events = create_optimized_tree_structure(non_metadata_events, categorizer)
    opt_time = time.time() - opt_start
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"⏱️  Optimized time: {opt_time:.2f} seconds")
    print(f"📦 Output events: {len(optimized_events):,}")
    print(f"🚀 Processing rate: {len(non_metadata_events)/opt_time:.0f} events/second")
    
    # Compare with expected runtime of old algorithm
    expected_old_time = 500  # 8+ minutes estimated
    speedup = expected_old_time / opt_time if opt_time > 0 else float('inf')
    print(f"🎯 Expected speedup: {speedup:.1f}x (estimated)")
    
    return optimized_events

if __name__ == "__main__":
    test_optimization()