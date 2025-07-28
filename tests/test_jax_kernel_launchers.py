#!/usr/bin/env python3
"""
Quick test script to verify JAX kernel launcher detection is working
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def test_kernel_launcher_detection():
    print("Testing JAX kernel launcher detection...")
    
    # Create analyzer from xplane.pb file
    print("Creating JAX TreePerfAnalyzer...")
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(
        "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    )
    
    print("Getting kernel launchers...")
    df_kernel_launchers = perf_analyzer.get_df_kernel_launchers(include_kernel_names=True)
    
    print(f"✅ SUCCESS: Found {len(df_kernel_launchers)} kernel launchers!")
    print(f"Columns: {list(df_kernel_launchers.columns)}")
    
    if len(df_kernel_launchers) > 0:
        print(f"First few kernel launchers:")
        print(df_kernel_launchers.head())
        return True
    else:
        print("❌ No kernel launchers found")
        return False

if __name__ == "__main__":
    test_kernel_launcher_detection()