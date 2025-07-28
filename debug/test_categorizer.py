#!/usr/bin/env python3
"""
Quick test for JAX categorizer fix
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.TreePerf.jax_analyses import JaxAnalyses

def test_categorizer():
    """Test the categorizer on synthetic events"""
    print("🧪 Testing JAX Categorizer Fix")
    print("=" * 40)
    
    # Create test events
    synthetic_runtime_event = {
        'name': 'cudaLaunchKernel',
        'cat': 'cuda_runtime',
        'ph': 'X',
        'pid': 0,
        'tid': 999999,
        'ts': 1000,
        'dur': 1
    }
    
    cpu_op_event = {
        'name': 'jit(train_step)',
        'ph': 'X', 
        'pid': 0,
        'tid': 123,
        'ts': 1000,
        'dur': 100
    }
    
    # Mock metadata
    metadata = {
        0: {
            123: {'thread_name': 'Framework Name Scope'},
            999999: {'thread_name': 'JAX_Synthetic_Runtime'}
        }
    }
    
    # Test categorization
    runtime_category = JaxAnalyses.get_event_category(metadata, synthetic_runtime_event)
    cpu_category = JaxAnalyses.get_event_category(metadata, cpu_op_event)
    
    print(f"🔍 Runtime event categorized as: '{runtime_category}'")
    print(f"🔍 CPU event categorized as: '{cpu_category}'")
    
    # Test expected outcomes
    assert runtime_category == "cuda_runtime", f"Expected 'cuda_runtime', got '{runtime_category}'"
    assert cpu_category == "cpu_op", f"Expected 'cpu_op', got '{cpu_category}'"
    
    print("✅ Categorizer fix working correctly!")
    return True

if __name__ == "__main__":
    test_categorizer()