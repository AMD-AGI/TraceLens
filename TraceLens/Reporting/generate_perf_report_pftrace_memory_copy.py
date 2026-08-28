###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Extract memory_copy operations from a Perfetto-style pftrace and report,
for each (copy_bytes, direction) the number of memory_copy operations (count).
Direction column indicates: h2d (which GPU), d2h (which GPU to host), d2d (which GPU -> which GPU).
Uses shared pftrace_utils (traceconv) and PftraceParser.
"""

import os
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import logging

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from TraceLens.util import PftraceParser
from TraceLens.Reporting.pftrace_utils import (
    derive_pftrace_output_path,
    ensure_trace_json,
)
from TraceLens.Reporting.reporting_utils import write_report_outputs

# Event name substrings for direction (ROCm/Perfetto)
NAME_HOST_TO_DEVICE = "MEMORY_COPY_HOST_TO_DEVICE"
NAME_DEVICE_TO_HOST = "MEMORY_COPY_DEVICE_TO_HOST"
NAME_DEVICE_TO_DEVICE = "MEMORY_COPY_DEVICE_TO_DEVICE"


def _get_copy_bytes(e: Dict[str, Any]) -> Optional[int]:
    """Extract copy_bytes from event args."""
    args = e.get("args") or {}
    val = args.get("copy_bytes")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _get_agent(args: Dict[str, Any], key: str) -> Optional[int]:
    """Get agent id (int) from args; key is 'src_agent' or 'dst_agent'."""
    val = args.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _format_direction(e: Dict[str, Any]) -> str:
    """
    Build direction string: h2d (GPU N), d2h (GPU N), d2d (GPU N -> GPU M).
    Assumes agent 0 = host when applicable; other agents = GPU index.
    """
    name = (e.get("name") or "").strip().upper()
    args = e.get("args") or {}
    src = _get_agent(args, "src_agent")
    dst = _get_agent(args, "dst_agent")

    if NAME_HOST_TO_DEVICE in name or name == NAME_HOST_TO_DEVICE:
        # Host -> Device: destination is the GPU
        gpu = dst if dst is not None else "?"
        return f"h2d (GPU {gpu})"
    if NAME_DEVICE_TO_HOST in name or name == NAME_DEVICE_TO_HOST:
        # Device -> Host: source is the GPU
        gpu = src if src is not None else "?"
        return f"d2h (GPU {gpu})"
    if NAME_DEVICE_TO_DEVICE in name or name == NAME_DEVICE_TO_DEVICE:
        # Device -> Device
        s = src if src is not None else "?"
        d = dst if dst is not None else "?"
        return f"d2d (GPU {s} -> GPU {d})"
    # Fallback: show raw name or unknown
    return "other"


def extract_memory_copy_rows(events: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    """
    From traceEvents, return list of (copy_bytes, direction) for every memory_copy
    event that has a valid copy_bytes. direction includes GPU info (e.g. h2d (GPU 2)).
    """
    out: List[Tuple[int, str]] = []
    for e in events:
        if (e.get("cat") or "").strip().lower() != "memory_copy":
            continue
        copy_bytes = _get_copy_bytes(e)
        if copy_bytes is None:
            continue
        direction = _format_direction(e)
        out.append((copy_bytes, direction))
    return out


def build_memory_copy_count_df(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a single DataFrame: for each (copy_bytes, direction), the number of
    memory_copy operations. Columns: copy_bytes, direction, count.
    """
    rows = extract_memory_copy_rows(events)
    if not rows:
        return pd.DataFrame(columns=["copy_bytes", "direction", "count"])

    df = pd.DataFrame(rows, columns=["copy_bytes", "direction"])
    count_df = df.groupby(["copy_bytes", "direction"], as_index=False).size()
    count_df = count_df.rename(columns={"size": "count"})
    return count_df


def generate_perf_report_pftrace_memory_copy(
    trace_path: str,
    output_xlsx_path: Optional[str] = None,
    output_csvs_dir: Optional[str] = None,
    traceconv_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load pftrace (or JSON), extract memory_copy operations, and return a table
    of each copy_bytes value with its occurrence count (number of ops with that size).

    Args:
        trace_path: Path to trace file (.json, .json.gz, or .pftrace).
        output_xlsx_path: Optional path to write Excel file.
        output_csvs_dir: Optional directory to write CSV files.
        traceconv_path: Optional path to traceconv for .pftrace conversion.

    Returns:
        Dict with key "memory_copy_by_copy_bytes": DataFrame with columns copy_bytes, direction, count.
    """
    logger.info("Loading trace from: %s", trace_path)
    json_path = ensure_trace_json(trace_path, traceconv_path)
    data = PftraceParser.load_pftrace_data(json_path)
    events = PftraceParser.get_events(data)

    count_df = build_memory_copy_count_df(events)
    dfs = {"memory_copy_by_copy_bytes": count_df}

    if not output_csvs_dir and output_xlsx_path is None:
        output_xlsx_path = derive_pftrace_output_path(
            trace_path, "_pftrace_memory_copy_report.xlsx"
        )
    write_report_outputs(
        dfs,
        xlsx_path=output_xlsx_path,
        csvs_dir=output_csvs_dir,
    )

    return dfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract memory_copy operations from pftrace; report count per copy_bytes (each size -> number of ops)."
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
        help="Directory to save output CSV files",
    )
    parser.add_argument(
        "--traceconv",
        type=str,
        default=None,
        dest="traceconv_path",
        help="Path to traceconv (optional; for .pftrace, auto-resolved from PATH or downloaded)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trace_path):
        logger.error("Input file not found: %s", args.trace_path)
        sys.exit(1)

    try:
        generate_perf_report_pftrace_memory_copy(
            trace_path=args.trace_path,
            output_xlsx_path=args.output_xlsx_path,
            output_csvs_dir=args.output_csvs_dir,
            traceconv_path=args.traceconv_path,
        )
    except Exception as e:
        logger.exception("Error generating memory copy report: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
