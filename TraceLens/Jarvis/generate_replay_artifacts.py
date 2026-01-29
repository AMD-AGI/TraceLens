#!/usr/bin/env python3
"""
Generate Replay Artifacts Script
Creates standalone replay packages for kernel team investigation
"""

import pandas as pd
import ast
import json
import zipfile
import os
import argparse
import sys

# Add TraceLens to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from TraceLens.EventReplay import EventReplayer, EventReplay


def row_to_evt(row):
    """Convert DataFrame row to event dict for replay"""
    return {
        'name': row['name'],
        'args': {
            'Input Dims': ast.literal_eval(row['Input Dims']) if isinstance(row['Input Dims'], str) else row['Input Dims'],
            'Input Strides': ast.literal_eval(row['Input Strides']) if isinstance(row['Input Strides'], str) else row['Input Strides'],
            'Input type': ast.literal_eval(row['Input type']) if isinstance(row['Input type'], str) else row['Input type'],
            'Concrete Inputs': ast.literal_eval(row['Concrete Inputs']) if isinstance(row['Concrete Inputs'], str) else row['Concrete Inputs'],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate replay artifacts for specified operations'
    )
    parser.add_argument('--output-dir', required=True, 
                       help='Output directory containing perf reports')
    parser.add_argument('--perf-report-path', required=True,
                       help='Path to perf_report.xlsx file')
    parser.add_argument('--op-names', nargs='+', required=True,
                       help='Operation names to generate replay artifacts for')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    perf_report_path = args.perf_report_path
    op_names = args.op_names
    
    print("=" * 80)
    print("GENERATE REPLAY ARTIFACTS")
    print("=" * 80)
    print(f"\nTarget operations: {', '.join(op_names)}")
    print(f"Perf report: {perf_report_path}")
    
    # Check if perf report exists
    if not os.path.exists(perf_report_path):
        print(f"\n❌ Error: Perf report not found: {perf_report_path}")
        return 1
    
    # Read unified perf summary
    try:
        df = pd.read_excel(perf_report_path, sheet_name='unified_perf_summary')
    except Exception as e:
        print(f"\n❌ Error reading perf report: {e}")
        return 1
    
    print(f"\nTotal operations in report: {len(df)}")
    
    # Create replay packages directory
    replay_dir = os.path.join(output_dir, 'replay_packages')
    os.makedirs(replay_dir, exist_ok=True)
    
    # Get TraceLens EventReplay module directory
    dir_replay = os.path.dirname(EventReplay.__file__)
    
    # Process each target operation
    total_created = 0
    for op_name in op_names:
        print(f"\n--- Processing: {op_name} ---")
        
        # Find matching operations
        target_ops = df[df['name'].str.contains(op_name, case=False, na=False)]
        
        if len(target_ops) == 0:
            print(f"  ⚠️  No operations found matching '{op_name}'")
            continue
        
        print(f"  Found {len(target_ops)} matching operations")
        
        # Generate replay configs
        repro_list = []
        for idx, row in target_ops.iterrows():
            try:
                replayer = EventReplayer(row_to_evt(row), lazy=True)
                repro_info = replayer.get_repro_info()
                repro_list.append(repro_info)
            except Exception as e:
                print(f"  ⚠️  Failed to create replay for row {idx}: {e}")
                continue
        
        if len(repro_list) == 0:
            print(f"  ❌ No replay configs generated for '{op_name}'")
            continue
        
        # Create sanitized filename
        safe_op_name = op_name.replace('::', '_').replace('/', '_').replace(' ', '_')
        replay_ir_file = os.path.join(replay_dir, f'{safe_op_name}_replay_ir.json')
        replay_package_file = os.path.join(replay_dir, f'{safe_op_name}_replay_package.zip')
        
        # Save replay IR
        with open(replay_ir_file, 'w') as f:
            json.dump(repro_list, f, indent=2)
        
        print(f"  ✓ Saved replay IR: {replay_ir_file}")
        
        # Create standalone package
        try:
            with zipfile.ZipFile(replay_package_file, 'w') as z:
                z.write(replay_ir_file, f'{safe_op_name}_replay_ir.json')
                
                # Add EventReplay utilities
                utils_path = os.path.join(dir_replay, 'utils.py')
                batched_replay_path = os.path.join(dir_replay, 'batched_replay.py')
                readme_path = os.path.join(dir_replay, 'batched_replay_readme.md')
                
                if os.path.exists(utils_path):
                    z.write(utils_path, 'utils.py')
                if os.path.exists(batched_replay_path):
                    z.write(batched_replay_path, 'batched_replay.py')
                if os.path.exists(readme_path):
                    z.write(readme_path, 'README.md')
            
            print(f"  ✓ Created replay package: {replay_package_file}")
            total_created += 1
            
        except Exception as e:
            print(f"  ❌ Failed to create replay package: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"✓ Replay artifact generation complete")
    print(f"  Created {total_created} replay packages in: {replay_dir}")
    print(f"{'=' * 80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
