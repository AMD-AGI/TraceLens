#!/usr/bin/env python3
"""Quick test to verify JAX integration is working"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("Quick JAX Integration Test")
    print("=" * 30)
    
    # Test 1: Basic imports
    try:
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxXplaneToTree, JaxTreePerfAnalyzer
        print("✅ JAX imports successful")
    except Exception as e:
        print(f"❌ JAX imports failed: {e}")
        return 1
    
    # Test 2: Tree creation
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    if not os.path.exists(xplane_path):
        print(f"❌ Test file not found: {xplane_path}")
        return 1
    
    try:
        print("Creating JAX tree...")
        tree = JaxXplaneToTree.from_xplane_pb(xplane_path)
        print(f"✅ Tree created with {len(tree.events):,} events")
    except Exception as e:
        print(f"❌ Tree creation failed: {e}")
        return 1
    
    # Test 3: TreePerfAnalyzer
    try:
        print("Creating TreePerfAnalyzer...")
        analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
        print(f"✅ Analyzer created (JAX mode: {analyzer.jax})")
    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")
        return 1
    
    # Test 4: Basic operations
    try:
        df_timeline = analyzer.get_df_gpu_timeline()
        print(f"✅ GPU timeline: {len(df_timeline)} rows")
        
        df_launchers = analyzer.get_df_kernel_launchers()
        print(f"✅ Kernel launchers: {len(df_launchers)} rows")
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return 1
    
    # Test 5: Dummy values
    try:
        from TraceLens.PerfModel.jax_op_mapping import JaxDummyPerfModel
        dummy = JaxDummyPerfModel({'name': 'test'})
        flops = dummy.flops()
        if flops < 0:
            print(f"✅ Dummy FLOPS working: {flops}")
        else:
            print(f"⚠️  Dummy FLOPS not negative: {flops}")
    except Exception as e:
        print(f"❌ Dummy values test failed: {e}")
        return 1
        
    print("\n🎉 All core functionality tests passed!")
    print("JAX integration is working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())