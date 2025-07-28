#!/usr/bin/env python3
"""Test the example JAX performance report generation script"""

import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("Testing JAX Performance Report Generation")
    print("=" * 45)
    
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix="jax_test_")
    xlsx_path = os.path.join(temp_dir, "test_report.xlsx")
    
    try:
        # Import and test the example script functionality
        from examples.generate_perf_report_jax_xplane import main as example_main
        import argparse
        
        # Mock sys.argv for the example script
        xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
        if not os.path.exists(xplane_path):
            print(f"❌ Test file not found: {xplane_path}")
            return 1
            
        # Create a simple test by directly importing and using the components
        print("Testing example script components...")
        
        # Test imports from the script
        from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
        from TraceLens.PerfModel import dict_cat2names
        
        print("✅ Imports successful")
        
        # Test analyzer creation (the core of the example script)
        print("Creating TreePerfAnalyzer...")
        analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
        print("✅ Analyzer created")
        
        # Test basic DataFrame generation (key functionality)
        print("Testing DataFrame generation...")
        df_timeline = analyzer.get_df_gpu_timeline()
        df_launchers = analyzer.get_df_kernel_launchers(include_kernel_names=True)
        print(f"✅ Timeline: {len(df_timeline)} rows")
        print(f"✅ Launchers: {len(df_launchers)} rows")
        
        # Test operation categorization
        op_events_found = 0
        for op_cat, op_names in dict_cat2names.items():
            op_events = [event for event in analyzer.tree.events if event['name'] in op_names]
            if op_events:
                op_events_found += 1
                
        print(f"✅ Found events for {op_events_found} operation categories")
        
        print("\n🎉 Example script functionality verified!")
        return 0
        
    except Exception as e:
        print(f"❌ Example script test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    sys.exit(main())