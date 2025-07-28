#!/usr/bin/env python3
"""
Test the complete fixed JAX pipeline end-to-end
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def test_fixed_jax_pipeline():
    """Test the complete fixed JAX pipeline"""
    print("🧪 Testing Fixed JAX Pipeline End-to-End")
    print("=" * 50)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    # Test kernel launcher detection (this was failing before)
    print("\n🔍 Testing get_kernel_launchers()...")
    try:
        kernel_launchers = perf_analyzer.get_kernel_launchers()
        print(f"✅ SUCCESS: Found {len(kernel_launchers)} kernel launchers")
        
        if len(kernel_launchers) > 0:
            first_launcher = kernel_launchers[0]
            print(f"   First launcher: {first_launcher.get('name', 'UNKNOWN')}")
            print(f"   UID: {first_launcher.get('UID', 'NO_UID')}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test GPU event analysis using get_df_kernel_launchers
    print("\n🔍 Testing GPU event analysis via get_df_kernel_launchers()...")
    try:
        df_kernel_launchers = perf_analyzer.get_df_kernel_launchers()
        print(f"✅ SUCCESS: Generated DataFrame with {len(df_kernel_launchers)} rows")
        
        if len(df_kernel_launchers) > 0:
            print(f"   DataFrame columns: {list(df_kernel_launchers.columns)}")
            print(f"   Sample total_direct_kernel_time: {df_kernel_launchers['total_direct_kernel_time'].iloc[0]:.2f}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests passed! JAX pipeline is working correctly.")
    return True

if __name__ == "__main__":
    test_fixed_jax_pipeline()