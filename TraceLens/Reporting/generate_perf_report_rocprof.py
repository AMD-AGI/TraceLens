###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import argparse
import sys
import subprocess
from typing import Optional, Dict
import pandas as pd

from TraceLens.util import RocprofParser
from TraceLens.Reporting.rocprof_analysis import RocprofAnalyzer


def request_install(package_name):
    choice = (
        input(f"Do you want to install '{package_name}' via pip? [y/N]: ")
        .strip()
        .lower()
    )
    if choice == "y":
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
        except subprocess.CalledProcessError:
            print(
                f"Failed to install '{package_name}'. Please install it manually. Exiting."
            )
            sys.exit(1)
    else:
        print(f"Skipping installation of '{package_name}' and exiting.")
        sys.exit(1)


def generate_perf_report_rocprof(
    profile_json_path: str,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    kernel_summary: bool = True,
    kernel_details: bool = False,
    short_kernel_study: bool = False,
    short_kernel_threshold_us: int = 10,
    short_kernel_histogram_bins: int = 100,
    topk_kernels: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Process rocprofv3 JSON profile and generate performance reports

    Args:
        profile_json_path: Path to *_results.json file from rocprofv3
        output_xlsx_path: Output Excel file path (optional)
        output_csvs_dir: Output directory for CSV files (optional)
        kernel_summary: Include detailed kernel summary sheet
        kernel_details: Include detailed kernel information with grid/block dims
        short_kernel_study: Analyze short kernels
        short_kernel_threshold_us: Threshold for short kernels (microseconds)
        short_kernel_histogram_bins: Number of bins for short kernel histogram
        topk_kernels: Limit kernel details to top K kernels by time

    Returns:
        Dictionary of DataFrames (sheet_name -> DataFrame)

    Example:
        >>> dfs = generate_perf_report_rocprof(
        ...     'trace_results.json',
        ...     output_xlsx_path='report.xlsx',
        ...     short_kernel_study=True
        ... )
    """

    print(f"Loading rocprofv3 data from: {profile_json_path}")

    # 1. Load and parse rocprof data
    try:
        rocprof_data = RocprofParser.load_rocprof_data(profile_json_path)
    except Exception as e:
        print(f"Error loading rocprof data: {e}")
        raise

    # 2. Extract events
    print("Extracting kernel events...")
    kernel_events = RocprofParser.extract_kernel_events(rocprof_data)
    print(f"  Found {len(kernel_events)} kernel dispatches")

    print("Extracting memory events...")
    memory_events = RocprofParser.extract_memory_events(rocprof_data)
    print(f"  Found {len(memory_events)} memory operations")

    print("Extracting API events...")
    api_events = RocprofParser.extract_api_events(rocprof_data)
    print(f"  Found {len(api_events)} API calls")

    metadata = RocprofParser.get_metadata(rocprof_data)
    print(f"  PID: {metadata.get('pid')}, Hostname: {metadata.get('hostname')}")

    # 3. Create analyzer
    print("\nGenerating performance analysis...")
    analyzer = RocprofAnalyzer(kernel_events, memory_events, api_events, metadata)

    # 4. Generate DataFrames
    dict_name2df = {}

    print("  - GPU timeline")
    dict_name2df["gpu_timeline"] = analyzer.get_df_gpu_timeline()

    if kernel_summary:
        print("  - Kernel summary")
        dict_name2df["kernel_summary"] = analyzer.get_df_kernel_summary()
        print("  - Kernel summary by category")
        dict_name2df["kernel_summary_by_category"] = analyzer.get_df_kernel_summary_by_category()

    if kernel_details:
        print("  - Kernel details")
        dict_name2df["kernel_details"] = analyzer.get_df_kernel_details(topk=topk_kernels)

    if short_kernel_study:
        print(f"  - Short kernels (threshold: {short_kernel_threshold_us} µs)")
        dict_name2df["short_kernels_summary"] = analyzer.get_df_short_kernels(short_kernel_threshold_us)
        dict_name2df["short_kernel_histogram"] = analyzer.get_df_short_kernel_histogram(
            short_kernel_threshold_us, short_kernel_histogram_bins
        )

    # 5. Write output
    if output_csvs_dir:
        print(f"\nWriting CSV files to: {output_csvs_dir}")
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  - {sheet_name}.csv ({len(df)} rows)")
    else:
        if output_xlsx_path is None:
            # Auto-generate output filename
            if profile_json_path.endswith('_results.json'):
                output_xlsx_path = profile_json_path.replace('_results.json', '_perf_report.xlsx')
            else:
                base_path = profile_json_path.rsplit('.json', 1)[0]
                output_xlsx_path = base_path + '_perf_report.xlsx'

        print(f"\nWriting Excel file to: {output_xlsx_path}")
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing openpyxl: {e}")
            request_install("openpyxl")

        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  - Sheet '{sheet_name}' ({len(df)} rows)")

        print(f"\nSuccessfully written to {output_xlsx_path}")

    return dict_name2df


def main():
    """Command-line interface for rocprofv3 performance report generation"""

    parser = argparse.ArgumentParser(
        description="Process rocprofv3 JSON profile and generate performance reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Excel report from rocprof results
  TraceLens_generate_perf_report_rocprof --profile_json_path trace_results.json

  # Generate with short kernel analysis
  TraceLens_generate_perf_report_rocprof --profile_json_path trace.json --short_kernel_study

  # Generate CSV files instead of Excel
  TraceLens_generate_perf_report_rocprof --profile_json_path trace.json --output_csvs_dir ./output

  # Custom output path with kernel details
  TraceLens_generate_perf_report_rocprof --profile_json_path trace.json \\
      --output_xlsx_path my_report.xlsx --kernel_details --topk_kernels 100
        """
    )

    # Required arguments
    parser.add_argument(
        "--profile_json_path",
        type=str,
        required=True,
        help="Path to the rocprofv3 *_results.json file",
    )

    # Output options
    parser.add_argument(
        "--output_xlsx_path",
        type=str,
        default=None,
        help="Path to the output Excel file",
    )
    parser.add_argument(
        "--output_csvs_dir",
        type=str,
        default=None,
        help="Directory to save output CSV files (alternative to Excel)",
    )

    # Analysis options
    parser.add_argument(
        "--disable_kernel_summary",
        action="store_false",
        dest="kernel_summary",
        default=True,
        help="Disable kernel summary sheets (enabled by default)",
    )
    parser.add_argument(
        "--kernel_details",
        action="store_true",
        help="Include detailed kernel information with grid/block dimensions",
    )
    parser.add_argument(
        "--short_kernel_study",
        action="store_true",
        help="Include short kernel analysis in the report",
    )
    parser.add_argument(
        "--short_kernel_threshold_us",
        type=int,
        default=10,
        help='Threshold in microseconds to classify a kernel as "short" (default: 10)',
    )
    parser.add_argument(
        "--short_kernel_histogram_bins",
        type=int,
        default=100,
        help="Number of bins for the short-kernel histogram (default: 100)",
    )
    parser.add_argument(
        "--topk_kernels",
        type=int,
        default=None,
        help="Limit kernel details to top K kernels by time (default: all)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.profile_json_path):
        print(f"Error: Input file not found: {args.profile_json_path}")
        sys.exit(1)

    # Generate report
    try:
        generate_perf_report_rocprof(
            profile_json_path=args.profile_json_path,
            output_xlsx_path=args.output_xlsx_path,
            output_csvs_dir=args.output_csvs_dir,
            kernel_summary=args.kernel_summary,
            kernel_details=args.kernel_details,
            short_kernel_study=args.short_kernel_study,
            short_kernel_threshold_us=args.short_kernel_threshold_us,
            short_kernel_histogram_bins=args.short_kernel_histogram_bins,
            topk_kernels=args.topk_kernels,
        )
    except Exception as e:
        print(f"\nError generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

