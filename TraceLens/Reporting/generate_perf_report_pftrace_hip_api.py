###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import re
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from TraceLens.util import PftraceParser
from TraceLens.Reporting.pftrace_utils import ensure_trace_json
from TraceLens.Reporting.pftrace_hip_api_analysis import PftraceHipApiAnalyzer


def generate_perf_report_pftrace_hip_api(
    trace_path: str,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    exclude_kernel_regex: Optional[str] = r"(?:^|\W)redzone_checker_kernel(?:\W|$)",
    allow_multi_kernel_per_api: bool = False,
    include_nonlaunch_apis: bool = False,
    traceconv_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Process Perfetto-style trace (traceEvents JSON or .pftrace) and generate
    HIP API ↔ Kernel performance summary report.

    This function serves dual purposes:
    1. CLI tool: Writes performance reports to Excel/CSV files
    2. Library function: Returns DataFrames for programmatic access and testing

    Args:
        trace_path: Path to trace file (.json, .json.gz, or .pftrace).
        output_xlsx_path: Output Excel file path (optional).
        output_csvs_dir: Output directory for CSV files (optional).
        exclude_kernel_regex: Regex to exclude kernel names from analysis (default: redzone_checker).
        allow_multi_kernel_per_api: If True, allow multiple kernels per API correlation ID.
        include_nonlaunch_apis: If True, include API rows with no linked kernel.
        traceconv_path: Optional path to Perfetto traceconv. For .pftrace, if not set,
            traceconv is looked up on PATH or downloaded into the trace file's directory.

    Returns:
        Dictionary mapping sheet names to DataFrames (e.g. "api_kernel_summary").

    Example:
        CLI usage:
            $ TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace_results.json

        Library usage:
            >>> dfs = generate_perf_report_pftrace_hip_api(
            ...     "trace_results.json",
            ...     output_xlsx_path="report.xlsx",
            ... )
            >>> list(dfs.keys())
            ['api_kernel_summary']
    """
    logger.info(f"Loading Perfetto-style trace from: {trace_path}")

    json_path = ensure_trace_json(trace_path, traceconv_path)

    try:
        data = PftraceParser.load_pftrace_data(json_path)
    except Exception as e:
        logger.error(f"Error loading trace data: {e}")
        raise

    events = PftraceParser.get_events(data)
    logger.info(f"  Found {len(events)} trace events")

    exclude_re = re.compile(exclude_kernel_regex) if exclude_kernel_regex else None
    analyzer = PftraceHipApiAnalyzer(
        events,
        exclude_kernel_re=exclude_re,
        allow_multi_kernel_per_api=allow_multi_kernel_per_api,
        include_nonlaunch_apis=include_nonlaunch_apis,
    )

    logger.info("Generating API ↔ Kernel summary...")
    dict_name2df = {}
    dict_name2df["api_kernel_summary"] = analyzer.get_df_api_kernel_summary()
    logger.info(
        f"  - api_kernel_summary ({len(dict_name2df['api_kernel_summary'])} rows)"
    )

    # Write output
    if output_csvs_dir:
        logger.info(f"Writing CSV files to: {output_csvs_dir}")
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"  - {sheet_name}.csv ({len(df)} rows)")
    else:
        if output_xlsx_path is None:
            base = Path(trace_path).resolve()
            if base.suffix.lower() == ".pftrace":
                base = base.with_suffix("")
            elif base.suffix.lower() == ".gz" and base.name.endswith(".json.gz"):
                base = base.parent / base.name.replace(".json.gz", "")
            else:
                base = base.with_suffix("")
            output_xlsx_path = str(base) + "_pftrace_hip_api_report.xlsx"

        logger.info(f"Writing Excel file to: {output_xlsx_path}")
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                f"Error importing openpyxl: {e}. Please install: pip install openpyxl"
            )
            raise

        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                logger.info(f"  - Sheet '{sheet_name}' ({len(df)} rows)")

        logger.info(f"Successfully written to {output_xlsx_path}")

    return dict_name2df


def main():
    """Command-line interface for HIP API ↔ Kernel report from Perfetto-style traces."""

    parser = argparse.ArgumentParser(
        description="Generate HIP API ↔ Kernel summary from Perfetto-style trace (traceEvents JSON or .pftrace)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Excel report from JSON trace
  TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace_results.json

  # Generate with .pftrace (traceconv auto-resolved from PATH or downloaded)
  TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace.pftrace
  # Or pass traceconv explicitly:
  TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace.pftrace --traceconv /path/to/traceconv

  # Output to CSV directory
  TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace.json --output_csvs_dir ./out

  # Include non-launch APIs and allow multiple kernels per API
  TraceLens_generate_perf_report_pftrace_hip_api --trace_path trace.json --include_nonlaunch_apis --allow_multi_kernel_per_api
        """,
    )

    parser.add_argument(
        "--trace_path",
        type=str,
        required=True,
        help="Path to trace file (.json, .json.gz, or .pftrace)",
    )
    parser.add_argument(
        "--output_xlsx_path",
        type=str,
        default=None,
        help="Path to output Excel file",
    )
    parser.add_argument(
        "--output_csvs_dir",
        type=str,
        default=None,
        help="Directory to save output CSV files (alternative to Excel)",
    )
    parser.add_argument(
        "--exclude_kernel_regex",
        type=str,
        default=r"(?:^|\W)redzone_checker_kernel(?:\W|$)",
        help="Regex to exclude kernel names from analysis",
    )
    parser.add_argument(
        "--allow_multi_kernel_per_api",
        action="store_true",
        help="Allow multiple kernels per API correlation ID",
    )
    parser.add_argument(
        "--include_nonlaunch_apis",
        action="store_true",
        help="Include API rows that have no linked kernel",
    )
    parser.add_argument(
        "--traceconv",
        type=str,
        default=None,
        dest="traceconv_path",
        help="Path to Perfetto traceconv (optional; for .pftrace, auto-resolved from PATH or downloaded)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.trace_path):
        logger.error(f"Input file not found: {args.trace_path}")
        sys.exit(1)

    try:
        generate_perf_report_pftrace_hip_api(
            trace_path=args.trace_path,
            output_xlsx_path=args.output_xlsx_path,
            output_csvs_dir=args.output_csvs_dir,
            exclude_kernel_regex=args.exclude_kernel_regex or None,
            allow_multi_kernel_per_api=args.allow_multi_kernel_per_api,
            include_nonlaunch_apis=args.include_nonlaunch_apis,
            traceconv_path=args.traceconv_path,
        )
    except Exception as e:
        logger.exception(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
