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

from . import perf_model
from collections import defaultdict


class JaxDummyPerfModel:
    """
    Dummy performance model for JAX operations that don't have specific implementations.
    
    This ensures TreePerfAnalyzer can process JAX events without errors while clearly
    indicating that the performance metrics are not real.
    """
    
    def __init__(self, event, arch=None, python_path=None):
        self.event = event
        self.arch = arch
        self.python_path = python_path
        
    def flops(self):
        """Return obvious dummy FLOPS value."""
        return -999999.0  # Negative to indicate dummy
        
    def flops_bwd(self):
        """Return obvious dummy backward FLOPS value.""" 
        return -888888.0  # Negative to indicate dummy
        
    def bytes(self):
        """Return obvious dummy bytes value."""
        return -777777  # Negative to indicate dummy
        
    def bytes_bwd(self):
        """Return obvious dummy backward bytes value."""
        return -666666  # Negative to indicate dummy


# JAX operation to performance model mapping
# Uses existing perf models where applicable, dummy models otherwise
jax_op_to_perf_model_class_map = {
    # GEMM operations - can use existing GEMM model from jax_analyses.py
    'custom-call': None,  # Will be handled specially by JaxAnalyses.get_perf_model
    
    # Common JAX operations that need dummy implementations
    'loop_convert_fusion': JaxDummyPerfModel,
    'input_reduce_fusion': JaxDummyPerfModel,
    'loop_slice_fusion': JaxDummyPerfModel,
    'wrapped_slice': JaxDummyPerfModel,
    'wrapped_transpose': JaxDummyPerfModel,
    'input_concatenate_fusion': JaxDummyPerfModel,
    'loop_multiply_fusion': JaxDummyPerfModel,
    'loop_add_fusion': JaxDummyPerfModel,
    'loop_rsqrt_fusion': JaxDummyPerfModel,
    'input_transpose_fusion': JaxDummyPerfModel,
    'loop_dynamic_slice_fusion': JaxDummyPerfModel,
    'loop_add_multiply_fusion': JaxDummyPerfModel,
    'loop_select_subtract_fusion': JaxDummyPerfModel,
    'input_convert_reduce_fusion': JaxDummyPerfModel,
    'gemm_fusion_dot': JaxDummyPerfModel,
    
    # Communication operations
    'all-gather-start': JaxDummyPerfModel,
    'all-reduce-start': JaxDummyPerfModel,
    'reduce-scatter': JaxDummyPerfModel,
    'collective-permute-start': JaxDummyPerfModel,
}


def get_jax_perf_model_class(event_name: str):
    """
    Get performance model class for a JAX event.
    
    Args:
        event_name: Name of the JAX event
        
    Returns:
        Performance model class or JaxDummyPerfModel if not found
    """
    # First check for exact matches
    if event_name in jax_op_to_perf_model_class_map:
        return jax_op_to_perf_model_class_map[event_name]
    
    # Check for partial matches (JAX operations often have numeric suffixes)
    base_name = event_name.split('.')[0] if '.' in event_name else event_name
    base_name = base_name.split('_')[0] + '_' + '_'.join(base_name.split('_')[1:3]) if '_' in base_name else base_name
    
    if base_name in jax_op_to_perf_model_class_map:
        return jax_op_to_perf_model_class_map[base_name]
    
    # For custom-call operations, check if it's a GEMM
    if 'custom-call' in event_name.lower():
        # Import JAX-specific logic only when needed
        try:
            from ..TreePerf.jax_analyses import JaxAnalyses
            # Create a dummy event to test with JaxAnalyses
            dummy_event = {'name': event_name}
            model_class = JaxAnalyses.get_perf_model(dummy_event)
            if model_class:
                return model_class
        except ImportError:
            pass
    
    # Default to dummy model
    return JaxDummyPerfModel


# Categories for JAX operations (similar to PyTorch dict_cat2names)
jax_dict_cat2names = defaultdict(list, {
    'GEMM': [
        'custom-call',  # Many GEMM operations appear as custom-call
        'gemm_fusion_dot',
    ],
    'Elementwise': [
        'loop_convert_fusion',
        'loop_multiply_fusion', 
        'loop_add_fusion',
        'loop_rsqrt_fusion',
        'loop_add_multiply_fusion',
        'loop_select_subtract_fusion',
    ],
    'Memory': [
        'wrapped_slice',
        'wrapped_transpose',
        'loop_slice_fusion',
        'loop_dynamic_slice_fusion',
        'input_concatenate_fusion',
        'input_transpose_fusion',
    ],
    'Reduction': [
        'input_reduce_fusion',
        'input_convert_reduce_fusion',
    ],
    'Communication': [
        'all-gather-start',
        'all-reduce-start', 
        'reduce-scatter',
        'collective-permute-start',
    ],
})