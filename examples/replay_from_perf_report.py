#!/usr/bin/env python
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Script to replay all operations from a TraceLens perf_report.xlsx spreadsheet.

Usage:
    python replay_from_perf_report.py /path/to/perf_report.xlsx [options]

Example:
    python replay_from_perf_report.py /data/jlichtne/EventReplay/ModelF-test/perf_report.xlsx --verbose
    python replay_from_perf_report.py perf_report.xlsx --op-filter "aten::mm" --device cuda
    python replay_from_perf_report.py perf_report.xlsx --sheet ops_unique_args --output replay_ir.json
"""

import argparse
import json
import ast
import os
import sys

# Add TraceLens to path if running from examples directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from TraceLens import EventReplayer
from TraceLens.EventReplay.event_replay import SkipReplayError


def row_to_event(row):
    """
    Convert a DataFrame row from the perf report to an event dict
    that EventReplayer can process.
    """
    event = {
        'name': row['name'],
        'args': {
            'Input Dims': ast.literal_eval(row['Input Dims']) if isinstance(row['Input Dims'], str) else row['Input Dims'],
            'Input Strides': ast.literal_eval(row['Input Strides']) if isinstance(row['Input Strides'], str) else row['Input Strides'],
            'Input type': ast.literal_eval(row['Input type']) if isinstance(row['Input type'], str) else row['Input type'],
            'Concrete Inputs': ast.literal_eval(row['Concrete Inputs']) if isinstance(row['Concrete Inputs'], str) else row['Concrete Inputs'],
        }
    }
    return event


def get_available_sheets(xlsx_path):
    """Get list of sheet names in the Excel file."""
    xl = pd.ExcelFile(xlsx_path)
    return xl.sheet_names


def find_ops_sheet(xlsx_path):
    """
    Find the appropriate sheet containing operation data.
    Looks for 'ops_unique_args' or 'kernel_launchers_unique_args'.
    """
    sheets = get_available_sheets(xlsx_path)
    
    # Preferred sheet names in order of preference
    preferred = ['ops_unique_args', 'kernel_launchers_unique_args']
    
    for sheet in preferred:
        if sheet in sheets:
            return sheet
    
    return None


def extract_replay_ir(xlsx_path, sheet_name=None, op_filter=None, verbose=False):
    """
    Extract replay IR from a perf report Excel file.
    
    Args:
        xlsx_path: Path to the perf_report.xlsx file
        sheet_name: Name of the sheet to read (auto-detected if None)
        op_filter: List of operation names to filter (None = all ops)
        verbose: Enable verbose output
        
    Returns:
        List of replay IR dictionaries
    """
    # Auto-detect sheet if not specified
    if sheet_name is None:
        sheet_name = find_ops_sheet(xlsx_path)
        if sheet_name is None:
            available = get_available_sheets(xlsx_path)
            raise ValueError(
                f"Could not find ops sheet. Available sheets: {available}\n"
                "Please specify --sheet argument."
            )
        print(f"Auto-detected sheet: '{sheet_name}'")
    
    # Read the sheet
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    
    # Check required columns exist
    required_cols = ['name', 'Input Dims', 'Input Strides', 'Input type', 'Concrete Inputs']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"Loaded {len(df)} operations from '{sheet_name}'")
    
    # Apply op filter if specified
    if op_filter:
        df = df[df['name'].isin(op_filter)].copy()
        print(f"After filtering for {op_filter}: {len(df)} operations")
    
    # Extract replay IR for each operation
    repro_data_list = []
    errors = []
    skipped = []

    for idx, row in df.iterrows():
        op_name = row['name']
        try:
            event = row_to_event(row)
            replayer = EventReplayer(event, lazy=True, verbose=verbose)
            repro_info = replayer.get_repro_info()

            # Optionally include count info for workload estimation
            if 'operation_count' in row:
                repro_info['count'] = int(row['operation_count'])

            repro_data_list.append(repro_info)

            if verbose:
                print(f"  Processed: {op_name}")

        except SkipReplayError as e:
            skipped.append((op_name, str(e)))
            if verbose:
                print(f"  Skipped: {op_name} ({e})")

        except Exception as e:
            errors.append((op_name, str(e)))
            if verbose:
                print(f"  Error processing {op_name}: {e}")

    print(f"\nSuccessfully extracted {len(repro_data_list)} operations")
    if skipped:
        print(f"Skipped (non-replayable): {len(skipped)}")
        for op_name, reason in skipped[:5]:
            print(f"  - {op_name}: {reason}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")
    if errors:
        print(f"Errors: {len(errors)}")
        for op_name, err in errors[:5]:  # Show first 5 errors
            print(f"  - {op_name}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    return repro_data_list


def run_batched_replay(replay_ir_path, device='cuda', verbose=False, stop_on_error=False):
    """
    Run the batched replay script on the extracted IR.
    """
    import subprocess
    from TraceLens import EventReplay
    
    dir_batched_replay = os.path.dirname(EventReplay.__file__)
    batched_replay_file = os.path.join(dir_batched_replay, "batched_replay.py")
    
    cmd = [
        sys.executable,
        batched_replay_file,
        replay_ir_path,
        "--device", device,
    ]
    
    if verbose:
        cmd.append("--verbose")
    if stop_on_error:
        cmd.append("--stop-on-error")
    
    print(f"\nRunning batched replay...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=dir_batched_replay, stdout=None, stderr=None)

    if result.returncode != 0:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Replay operations from a TraceLens perf_report.xlsx spreadsheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay all operations from a perf report
  python replay_from_perf_report.py /path/to/perf_report.xlsx

  # Replay only specific operations
  python replay_from_perf_report.py perf_report.xlsx --op-filter "aten::mm" "aten::addmm"

  # Just extract IR without replaying (for sharing)
  python replay_from_perf_report.py perf_report.xlsx --extract-only --output my_ops.json

  # Specify a different sheet name
  python replay_from_perf_report.py perf_report.xlsx --sheet kernel_launchers_unique_args
        """
    )
    
    parser.add_argument(
        "xlsx_path",
        type=str,
        help="Path to the perf_report.xlsx file"
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Sheet name to read (default: auto-detect 'ops_unique_args' or 'kernel_launchers_unique_args')"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="event_replay_ir.json",
        help="Output path for the replay IR JSON file (default: event_replay_ir.json)"
    )
    parser.add_argument(
        "--op-filter",
        nargs="+",
        default=None,
        help="Filter to specific operation names (e.g., 'aten::mm' 'aten::addmm')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run replay on (default: cuda)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract IR, don't run replay"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop if any operation fails during replay"
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="List available sheets and exit"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.xlsx_path):
        print(f"Error: File not found: {args.xlsx_path}")
        sys.exit(1)
    
    # List sheets mode
    if args.list_sheets:
        sheets = get_available_sheets(args.xlsx_path)
        print(f"Available sheets in '{args.xlsx_path}':")
        for sheet in sheets:
            print(f"  - {sheet}")
        sys.exit(0)
    
    print(f"Processing: {args.xlsx_path}")
    
    # Extract replay IR
    try:
        repro_data_list = extract_replay_ir(
            args.xlsx_path,
            sheet_name=args.sheet,
            op_filter=args.op_filter,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error extracting replay IR: {e}")
        sys.exit(1)
    
    if not repro_data_list:
        print("No operations to replay!")
        sys.exit(1)
    
    # Save the IR
    output_path = os.path.abspath(args.output)
    with open(output_path, 'w') as f:
        json.dump(repro_data_list, f, indent=4)
    print(f"\nSaved replay IR to: {output_path}")
    
    # Run replay unless extract-only
    if not args.extract_only:
        try:
            import torch
            if args.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                args.device = "cpu"
        except ImportError:
            print("Error: PyTorch is required for replay. Please install it first.")
            sys.exit(1)
        
        success = run_batched_replay(
            output_path,
            device=args.device,
            verbose=args.verbose,
            stop_on_error=args.stop_on_error
        )
        
        if not success:
            sys.exit(1)
    else:
        print("\n--extract-only specified. Skipping replay.")
        print(f"To run replay later: python batched_replay.py {output_path}")


if __name__ == "__main__":
    main()
