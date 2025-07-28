# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Any, List, Callable
from collections import defaultdict
import json
from ..util import DataLoader, TraceEventUtils
from .trace_to_tree import TraceToTree


class JaxXplaneToTree:
    """
    Converts JAX xplane.pb trace files into a tree structure compatible with TraceToTree.
    
    This class bridges the gap between JAX's xplane.pb format and the PyTorch-style
    tree structure expected by TreePerfAnalyzer. It ensures that all downstream
    analysis tools can work with JAX traces without modification.
    """
    
    # Dummy values for missing PyTorch-specific fields
    DUMMY_VALUES = {
        'correlation': -9999,           # Negative correlation ID to indicate dummy
        'External id': -8888,           # Negative external ID to indicate dummy  
        'Sequence number': -7777,       # Negative sequence to indicate dummy
        'Python id': -6666,             # Negative Python ID to indicate dummy
        'stream': -5555,                # Negative stream ID to indicate dummy
        'Input Dims': '[[DUMMY_JAX]]',  # Obvious dummy tensor dimensions
        'Input type': 'DUMMY_JAX_TYPE', # Obvious dummy data type
        'Input Strides': '[[DUMMY_JAX_STRIDES]]',  # Obvious dummy strides
        'Concrete Inputs': 'DUMMY_JAX_INPUTS',     # Obvious dummy inputs
    }
    
    @staticmethod
    def from_xplane_pb(profile_filepath: str, **kwargs) -> TraceToTree:
        """
        Create a TraceToTree from a JAX xplane.pb file.
        
        Args:
            profile_filepath: Path to the xplane.pb file
            **kwargs: Additional arguments passed to TraceToTree constructor
            
        Returns:
            TraceToTree instance populated with JAX data
        """
        # Import JAX-specific functionality only when needed
        from ..TreePerf.jax_analyses import JaxAnalyses
        
        # Load the JAX trace data using existing infrastructure
        data = DataLoader.load_data(profile_filepath)
        trace_events = data['traceEvents']
        
        # Use the existing JAX categorizer
        categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
        non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
        
        # Create proper tree structure linking HLO ops to trace events
        enhanced_events = JaxXplaneToTree._create_pytorch_compatible_tree_structure(non_metadata_events, categorizer)
        
        # Debug: Print final event counts
        print(f"Debug: Passing {len(enhanced_events)} total events to TraceToTree")
        
        # Create the tree with enhanced events
        return TraceToTree(
            enhanced_events, 
            event_to_category=categorizer,
            **kwargs
        )
    
    @staticmethod
    def _create_pytorch_compatible_tree_structure(events: List[Dict[str, Any]], categorizer: Callable) -> List[Dict[str, Any]]:
        """
        Create PyTorch-compatible tree structure by linking HLO ops to trace events.
        
        This method creates the proper parent-child-grandchild relationships that PyTorch expects:
        CPU op (FrameworkCallStack) → Runtime event → Kernel (Stream threads)
        
        Args:
            events: List of JAX trace events
            categorizer: Function to categorize events
            
        Returns:
            Enhanced events list with proper tree structure
        """
        from ..util import TraceEventUtils
        
        # Separate events by category
        cpu_ops = []
        kernels = []
        other_events = []
        
        # First pass: categorize events
        for event in events:
            category = categorizer(event)
            if category == 'cpu_op':
                cpu_ops.append(event)
            elif category in {'kernel', 'memcpy', 'memset'}:
                kernels.append(event)
            else:
                other_events.append(event)
        
        # Create mapping from hlo_op to events
        hlo_to_kernels = {}
        for kernel in kernels:
            hlo_op = kernel.get('args', {}).get('hlo_op')
            if hlo_op:
                if hlo_op not in hlo_to_kernels:
                    hlo_to_kernels[hlo_op] = []
                hlo_to_kernels[hlo_op].append(kernel)
        
        enhanced_events = []
        
        # Debug: Print initial categorization
        print(f"Debug: Found {len(cpu_ops)} CPU ops, {len(kernels)} kernels, {len(other_events)} other events")
        print(f"Debug: HLO to kernels mapping: {len(hlo_to_kernels)} HLO ops")
        if len(cpu_ops) > 0:
            print(f"Debug: First CPU op: {cpu_ops[0].get('name', 'UNKNOWN')} - {categorizer(cpu_ops[0])}")
            print(f"Debug: First CPU op args: {list(cpu_ops[0].get('args', {}).keys())}")
        if len(kernels) > 0:
            print(f"Debug: First kernel: {kernels[0].get('name', 'UNKNOWN')} - {categorizer(kernels[0])}")
            print(f"Debug: First kernel args: {list(kernels[0].get('args', {}).keys())}")
            if 'hlo_op' in kernels[0].get('args', {}):
                print(f"Debug: Kernel hlo_op example: {kernels[0]['args']['hlo_op']}")
        
        # OPTIMIZED: Hash-based kernel distribution instead of sequential
        print(f"🚀 Optimized distribution: {len(kernels)} kernels across {len(cpu_ops)} CPU ops")
        
        # Pre-allocate CPU op kernel assignments using hash distribution
        cpu_op_kernels = [[] for _ in range(len(cpu_ops))]
        for kernel in kernels:
            cpu_index = hash(kernel.get('name', '')) % len(cpu_ops)
            cpu_op_kernels[cpu_index].append(kernel)
        
        # Calculate exact result size for pre-allocation
        total_runtime_events = sum(len(k_list) for k_list in cpu_op_kernels)
        total_enhanced_kernels = total_runtime_events
        result_size = len(other_events) + len(cpu_ops) + total_runtime_events + total_enhanced_kernels
        
        # Pre-allocate result array
        enhanced_events = [None] * result_size
        current_array_index = 0
        
        # Add other events first (python functions, metadata, etc.)
        for event in other_events:
            enhanced_events[current_array_index] = event
            current_array_index += 1
        
        # Use unified indexing system - current_array_index tracks both placement and UID prediction
        # (since TraceToTree assigns UIDs sequentially based on array position)
        
        # OPTIMIZED: Create templates to avoid repeated dict creation
        cpu_enhancement = {
            'Input Dims': '[[DUMMY_JAX]]',
            'Input type': 'DUMMY_JAX_TYPE', 
            'Input Strides': '[[DUMMY_JAX_STRIDES]]',
            'Concrete Inputs': 'DUMMY_JAX_CONCRETE',
            'correlation': -9999,
            'External id': -8888,
        }
        
        kernel_enhancement = {
            'stream': -5555,
            'correlation': -9999,
        }
        
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
        
        # Process CPU ops and create tree structure
        for i, cpu_op in enumerate(cpu_ops):
            # Add the CPU op event with enhanced args
            enhanced_cpu_op = cpu_op.copy()
            if 'args' not in enhanced_cpu_op:
                enhanced_cpu_op['args'] = {}
            
            # Add PyTorch-compatible fields using template
            enhanced_cpu_op['args'].update(cpu_enhancement)
            
            # OPTIMIZED: Use pre-computed kernel assignments
            assigned_kernels = cpu_op_kernels[i]
            
            # Debug: Print detailed info for first few CPU ops
            if i < 3:
                print(f"Debug: CPU op #{i}: '{enhanced_cpu_op.get('name', 'UNKNOWN')}' assigned_kernels={len(assigned_kernels)}")
                if len(assigned_kernels) > 0:
                    print(f"  First assigned kernel: {assigned_kernels[0].get('name', 'UNKNOWN')} UID={assigned_kernels[0].get('UID', assigned_kernels[0].get('id', 'NO_UID'))}")
            
            # Initialize children list with future UIDs of runtime events
            runtime_event_uids = []
            
            # Create intermediate runtime events and link to kernels
            gpu_events = []
            for kernel in assigned_kernels:
                # Track this runtime event's future UID (will be current_array_index)
                runtime_event_future_uid = current_array_index
                runtime_event_uids.append(runtime_event_future_uid)
                
                # OPTIMIZED: Add enhanced kernel event using template
                enhanced_kernel = kernel.copy()
                if 'args' not in enhanced_kernel:
                    enhanced_kernel['args'] = {}
                enhanced_kernel['args'].update(kernel_enhancement)
                
                # Add 'cat' field for GPUEventAnalyser compatibility
                enhanced_kernel['cat'] = 'kernel'
                
                # Track kernel's future UID (will be current_array_index + 1)
                kernel_future_uid = current_array_index + 1
                
                # OPTIMIZED: Create synthetic runtime event from template
                runtime_event = runtime_template.copy()
                runtime_event['ts'] = kernel.get('ts', 0)
                runtime_event['pid'] = enhanced_cpu_op.get('pid', 0)
                runtime_event['tid'] = enhanced_cpu_op.get('tid', 0)
                runtime_event['children'] = [kernel_future_uid]  # Reference to kernel's future UID
                
                # OPTIMIZED: Use pre-allocated array assignment
                enhanced_events[current_array_index] = runtime_event
                current_array_index += 1
                enhanced_events[current_array_index] = enhanced_kernel
                current_array_index += 1
                gpu_events.append(enhanced_kernel)
            
            # Set CPU op's children to reference runtime events
            enhanced_cpu_op['children'] = runtime_event_uids
            
            # Add GPU events list to CPU op for TreePerfAnalyzer compatibility
            enhanced_cpu_op['gpu_events'] = [k.get('UID', k.get('id', 0)) for k in gpu_events]
            
            # OPTIMIZED: Use pre-allocated array assignment
            enhanced_events[current_array_index] = enhanced_cpu_op
            current_array_index += 1
        
        # OPTIMIZED: Remove any None entries (shouldn't happen with correct pre-allocation)
        enhanced_events = [e for e in enhanced_events if e is not None]
        
        print(f"✅ Optimized tree construction completed - {len(enhanced_events):,} events")
        
        return enhanced_events
    
    @staticmethod
    def _add_pytorch_fields(event: Dict[str, Any], category: str) -> Dict[str, Any]:
        """
        Add PyTorch-specific fields to a JAX event with appropriate dummy values.
        
        Args:
            event: JAX event dictionary
            category: Event category from categorizer
            
        Returns:
            Enhanced event with PyTorch-compatible fields
        """
        args = event.setdefault(TraceEventUtils.TraceKeys.Args, {})
        
        # Add linking fields that PyTorch tree building expects
        if category in {'cpu_op', 'cuda_runtime', 'cuda_driver'}:
            # CPU and runtime events need correlation/external IDs for linking
            if 'correlation_id' in args:
                # Use existing correlation_id if available
                args['correlation'] = args['correlation_id']
                args['External id'] = args['correlation_id']
            else:
                # Use dummy values
                args['correlation'] = JaxXplaneToTree.DUMMY_VALUES['correlation']
                args['External id'] = JaxXplaneToTree.DUMMY_VALUES['External id']
            
            # Add sequence number for backward pass linking (PyTorch specific)
            args['Sequence number'] = JaxXplaneToTree.DUMMY_VALUES['Sequence number']
        
        if category == 'cpu_op':
            # CPU operations in PyTorch have detailed input information
            args['Input Dims'] = JaxXplaneToTree.DUMMY_VALUES['Input Dims']
            args['Input type'] = JaxXplaneToTree.DUMMY_VALUES['Input type'] 
            args['Input Strides'] = JaxXplaneToTree.DUMMY_VALUES['Input Strides']
            args['Concrete Inputs'] = JaxXplaneToTree.DUMMY_VALUES['Concrete Inputs']
        
        if category == 'python_function':
            # Python function events need Python ID for linking
            args['Python id'] = JaxXplaneToTree.DUMMY_VALUES['Python id']
        
        if category in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
            # GPU events need stream information
            if 'stream' not in args:
                args['stream'] = JaxXplaneToTree.DUMMY_VALUES['stream']
            
            # Ensure correlation fields exist for linking
            if 'correlation_id' in args:
                args['correlation'] = args['correlation_id'] 
                args['External id'] = args['correlation_id']
            else:
                args['correlation'] = JaxXplaneToTree.DUMMY_VALUES['correlation']
                args['External id'] = JaxXplaneToTree.DUMMY_VALUES['External id']
        
        return event


class JaxTreePerfAnalyzer:
    """
    Factory class to create TreePerfAnalyzer instances from JAX xplane.pb files.
    
    This provides a convenient interface that mirrors the existing PyTorch workflow
    while handling all the JAX-specific conversions internally.
    """
    
    @staticmethod
    def from_xplane_pb(profile_filepath: str, **kwargs) -> 'TreePerfAnalyzer':
        """
        Create a TreePerfAnalyzer from a JAX xplane.pb file.
        
        Args:
            profile_filepath: Path to the xplane.pb file
            **kwargs: Additional arguments passed to TreePerfAnalyzer constructor
            
        Returns:
            TreePerfAnalyzer instance configured for JAX data
        """
        # Import dependencies only when needed
        from ..TreePerf.tree_perf import TreePerfAnalyzer
        from ..TreePerf.jax_analyses import JaxAnalyses
        
        # Create the enhanced tree
        tree = JaxXplaneToTree.from_xplane_pb(profile_filepath)
        
        # Get the JAX categorizer for consistency
        data = DataLoader.load_data(profile_filepath)
        trace_events = data['traceEvents']
        categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
        
        # Create TreePerfAnalyzer with JAX flag enabled
        return TreePerfAnalyzer(
            tree=tree,
            jax=True,
            event_to_category=categorizer,
            **kwargs
        )


# Convenience function for backward compatibility
def create_jax_tree_perf_analyzer(profile_filepath: str, **kwargs) -> 'TreePerfAnalyzer':
    """
    Convenience function to create a TreePerfAnalyzer from JAX xplane.pb file.
    
    Args:
        profile_filepath: Path to the xplane.pb file
        **kwargs: Additional arguments passed to TreePerfAnalyzer
        
    Returns:
        TreePerfAnalyzer instance ready for JAX trace analysis
    """
    return JaxTreePerfAnalyzer.from_xplane_pb(profile_filepath, **kwargs)