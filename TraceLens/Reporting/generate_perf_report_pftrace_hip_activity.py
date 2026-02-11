###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Generate ROCm Perfetto trace (pftrace) activity report: category summary per GPU,
kernel summary (NSYS-like), HIP API summary, XLA top kernels. Uses shared
pftrace_utils (traceconv) and PftraceHipActivityAnalyzer; does not duplicate
API↔kernel correlation (see generate_perf_report_pftrace_hip_api for that).
"""

import os
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from TraceLens.util import PftraceParser
from TraceLens.Reporting.pftrace_utils import ensure_trace_json
from TraceLens.Reporting.pftrace_hip_activity_analysis import (
    PftraceHipActivityAnalyzer,
    ns_to_ms,
)


def _write_markdown_report(
    out_path: Path,
    df_category: pd.DataFrame,
    xla_top: list,
    used_fav3: bool,
    agents: list,
    kernel_df: Optional[pd.DataFrame] = None,
    kernel_top: int = 200,
    hip_df: Optional[pd.DataFrame] = None,
    hip_top: int = 200,
) -> None:
    """Write a Markdown report. Uses pandas to_markdown if tabulate available."""
    lines = [
        "# ROCm Perfetto Trace Report",
        "",
        f"- Agents (GPUs): {', '.join(agents)}",
        f"- Used FA v3 kernels detected: **{1 if used_fav3 else 0}**",
        "",
        "## Category Summary (per GPU)",
    ]
    if df_category is not None and not df_category.empty:
        try:
            lines.append(df_category.to_markdown(index=False))
        except AttributeError:
            lines.append(df_category.to_string(index=False))
    else:
        lines.append("_(no data)_")
    lines.append("")
    lines.append("## Top XLA Kernels (by time)")
    if not xla_top:
        lines.append("_(no data)_")
    else:
        lines.append("| Kernel | Total Time (ms) | Count | Fraction of XLA |")
        lines.append("|---|---:|---:|---:|")
        for name, tot_ns, cnt, frac in xla_top:
            lines.append(f"| `{name}` | {ns_to_ms(tot_ns):.2f} | {cnt} | {frac:.2%} |")
    lines.append("")
    if kernel_df is not None and not kernel_df.empty:
        lines.append("### Kernel Summary Table")
        try:
            lines.append(kernel_df.head(kernel_top).to_markdown(index=False))
        except AttributeError:
            lines.append(kernel_df.head(kernel_top).to_string(index=False))
        lines.append("")
    if hip_df is not None and not hip_df.empty:
        lines.append("### HIP API Summary Table")
        try:
            lines.append(hip_df.head(hip_top).to_markdown(index=False))
        except AttributeError:
            lines.append(hip_df.head(hip_top).to_string(index=False))
        lines.append("")
    out_path.write_text("\n".join(lines))


def generate_perf_report_pftrace_hip_activity(
    trace_path: str,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    output_md_path: Optional[str] = None,
    traceconv_path: Optional[str] = None,
    merge_kernels: bool = False,
    min_tid: int = -10**9,
    max_tid: int = 10**9,
    min_event_ns: int = 5000,
    kernel_summary: bool = True,
    kernel_summary_include_rccl: bool = False,
    kernel_summary_baseline: str = "total",
    kernel_summary_group: str = "config",
    hip_summary: bool = True,
    hip_summary_group: str = "name",
    top_xla: int = 30,
    kernel_summary_top: int = 200,
    hip_summary_top: int = 200,
) -> Dict[str, pd.DataFrame]:
    """
    Process Perfetto-style trace (.json, .json.gz, or .pftrace) and generate
    GPU category summary, kernel summary, HIP API summary, XLA top (NSYS-like).

    Returns:
        Dict of sheet name -> DataFrame (category_summary, xla_top, kernel_summary,
        hip_summary when enabled). Also writes Excel/CSV and optionally Markdown.
    """
    logger.info("Loading Perfetto-style trace from: %s", trace_path)
    json_path = ensure_trace_json(trace_path, traceconv_path)
    try:
        data = PftraceParser.load_pftrace_data(json_path)
    except Exception as e:
        logger.error("Error loading trace: %s", e)
        raise
    events = PftraceParser.get_events(data)
    logger.info("  Found %d trace events", len(events))

    analyzer = PftraceHipActivityAnalyzer(
        events,
        merge_kernels=merge_kernels,
        min_tid=min_tid,
        max_tid=max_tid,
        min_event_ns=min_event_ns,
        kernel_summary_include_rccl=kernel_summary_include_rccl,
        kernel_summary_baseline=kernel_summary_baseline,
        kernel_summary_group=kernel_summary_group,
        hip_summary_group=hip_summary_group,
    )
    logger.info("  Detected GPUs (agents): %s", analyzer.agents)

    dict_name2df = {}
    dict_name2df["category_summary"] = analyzer.get_df_category_summary()
    logger.info("  - category_summary (%d rows)", len(dict_name2df["category_summary"]))
    dict_name2df["xla_top"] = analyzer.get_df_xla_top(top_n=top_xla)
    logger.info("  - xla_top (%d rows)", len(dict_name2df["xla_top"]))

    if kernel_summary:
        dict_name2df["kernel_summary"] = analyzer.get_df_kernel_summary()
        logger.info("  - kernel_summary (%d rows)", len(dict_name2df["kernel_summary"]))
    if hip_summary:
        dict_name2df["hip_summary"] = analyzer.get_df_hip_summary()
        logger.info("  - hip_summary (%d rows)", len(dict_name2df["hip_summary"]))

    if output_csvs_dir:
        logger.info("Writing CSV files to: %s", output_csvs_dir)
        os.makedirs(output_csvs_dir, exist_ok=True)
        for sheet_name, df in dict_name2df.items():
            csv_path = os.path.join(output_csvs_dir, f"{sheet_name}.csv")
            df.to_csv(csv_path, index=False)
            logger.info("  - %s.csv (%d rows)", sheet_name, len(df))
    else:
        if output_xlsx_path is None:
            base = Path(trace_path).resolve()
            if base.suffix.lower() == ".pftrace":
                base = base.with_suffix("")
            elif base.suffix.lower() == ".gz" and base.name.endswith(".json.gz"):
                base = base.parent / base.name.replace(".json.gz", "")
            else:
                base = base.with_suffix("")
            output_xlsx_path = str(base) + "_pftrace_activity_report.xlsx"
        logger.info("Writing Excel to: %s", output_xlsx_path)
        try:
            import openpyxl
        except (ImportError, ModuleNotFoundError) as e:
            logger.error("openpyxl required: %s", e)
            raise
        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            for sheet_name, df in dict_name2df.items():
                sn = sheet_name[:31]
                df.to_excel(writer, sheet_name=sn, index=False)
                logger.info("  - Sheet '%s' (%d rows)", sn, len(df))
        logger.info("Successfully written to %s", output_xlsx_path)

    if output_md_path:
        logger.info("Writing Markdown to: %s", output_md_path)
        _write_markdown_report(
            Path(output_md_path),
            df_category=dict_name2df["category_summary"],
            xla_top=analyzer.get_xla_top(top_n=top_xla),
            used_fav3=analyzer.used_fav3,
            agents=analyzer.agents,
            kernel_df=dict_name2df.get("kernel_summary"),
            kernel_top=kernel_summary_top,
            hip_df=dict_name2df.get("hip_summary"),
            hip_top=hip_summary_top,
        )

    return dict_name2df


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROCm Perfetto trace activity report (category, kernel, HIP API summaries)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  TraceLens_generate_perf_report_pftrace_hip_activity --trace_path trace.json
  TraceLens_generate_perf_report_pftrace_hip_activity --trace_path trace.pftrace --write_md
  TraceLens_generate_perf_report_pftrace_hip_activity --trace_path trace.json --output_csvs_dir ./out
        """,
    )
    parser.add_argument("--trace_path", type=str, required=True, help="Path to .json / .json.gz / .pftrace")
    parser.add_argument("--output_xlsx_path", type=str, default=None)
    parser.add_argument("--output_csvs_dir", type=str, default=None)
    parser.add_argument("--output_md_path", type=str, default=None, help="Write Markdown report to this path")
    parser.add_argument("--traceconv", type=str, default=None, dest="traceconv_path")
    parser.add_argument("--merge_kernels", action="store_true", help="Merge kernel names by stripping digits")
    parser.add_argument("--min_tid", type=int, default=-10**9)
    parser.add_argument("--max_tid", type=int, default=10**9)
    parser.add_argument("--min_event_ns", type=int, default=5000, help="Drop events shorter than this (ns)")
    parser.add_argument("--no_kernel_summary", action="store_false", dest="kernel_summary")
    parser.add_argument("--kernel_summary_include_rccl", action="store_true")
    parser.add_argument("--kernel_summary_baseline", choices=["total", "compute"], default="total")
    parser.add_argument("--kernel_summary_group", choices=["config", "name"], default="config")
    parser.add_argument("--no_hip_summary", action="store_false", dest="hip_summary")
    parser.add_argument("--hip_summary_group", choices=["name", "name+stream", "name+op", "name+stream+op"], default="name")
    parser.add_argument("--top_xla", type=int, default=30)
    parser.add_argument("--kernel_summary_top", type=int, default=200)
    parser.add_argument("--hip_summary_top", type=int, default=200)
    parser.add_argument("--write_md", action="store_true", help="Write Markdown report (default path: <trace_stem>_report.md)")

    args = parser.parse_args()
    if not os.path.exists(args.trace_path):
        logger.error("Input file not found: %s", args.trace_path)
        sys.exit(1)

    md_path = args.output_md_path
    if args.write_md and md_path is None:
        base = Path(args.trace_path).resolve().stem
        if base.endswith(".json"):
            base = base[:-5]
        md_path = str(Path(args.trace_path).resolve().parent / f"{base}_report.md")

    try:
        generate_perf_report_pftrace_hip_activity(
            trace_path=args.trace_path,
            output_xlsx_path=args.output_xlsx_path,
            output_csvs_dir=args.output_csvs_dir,
            output_md_path=md_path,
            traceconv_path=args.traceconv_path,
            merge_kernels=args.merge_kernels,
            min_tid=args.min_tid,
            max_tid=args.max_tid,
            min_event_ns=args.min_event_ns,
            kernel_summary=args.kernel_summary,
            kernel_summary_include_rccl=args.kernel_summary_include_rccl,
            kernel_summary_baseline=args.kernel_summary_baseline,
            kernel_summary_group=args.kernel_summary_group,
            hip_summary=args.hip_summary,
            hip_summary_group=args.hip_summary_group,
            top_xla=args.top_xla,
            kernel_summary_top=args.kernel_summary_top,
            hip_summary_top=args.hip_summary_top,
        )
    except Exception as e:
        logger.exception("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
