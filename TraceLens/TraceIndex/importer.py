###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend-neutral TraceLens report import workflow."""

import re
from pathlib import Path
from typing import Optional

from TraceLens.TraceIndex.models import TraceRecord, TraceReport
from TraceLens.TraceIndex.scanner import trace_record_from_path
from TraceLens.TraceIndex.store import TraceIndexStore
from TraceLens.TraceIndex.utils import normalize_path, read_csv_rows


REPORT_SHEETS = (
    "unified_perf_summary",
    "kernel_summary",
    "ops_summary_by_category",
    "gpu_timeline",
)


def load_report_dir(report_dir: Path) -> TraceReport:
    report_dir = report_dir.resolve()
    return TraceReport(
        report_dir=normalize_path(report_dir),
        sheets={
            sheet_name: read_csv_rows(report_dir / ("%s.csv" % sheet_name))
            for sheet_name in REPORT_SHEETS
        },
    )


def synthetic_trace_record_for_report(report_dir: Path) -> TraceRecord:
    report_dir = report_dir.resolve()
    return TraceRecord(
        root=None,
        path=normalize_path(report_dir),
        rel_path=report_dir.name,
        name=report_dir.name,
        size_bytes=None,
        md5=None,
        format="tracelens_report_dir",
        rank=None,
        top_dir=None,
        parent_rel=None,
        should_enrich=True,
        skip_reason=None,
    )


def import_report_dir(
    store: TraceIndexStore,
    report_dir: Path,
    trace_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> int:
    store.init_schema()
    trace = (
        trace_record_from_path(trace_path, root=root)
        if trace_path is not None
        else synthetic_trace_record_for_report(report_dir)
    )
    trace_id = store.upsert_trace(trace)
    store.import_report(trace_id, load_report_dir(report_dir))
    return trace_id


def generate_report_and_import(
    store: TraceIndexStore,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> int:
    if report_dir is None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_path.name)
        report_dir = Path("trace_index_reports") / safe_name
    report_dir = report_dir.resolve()
    unified_csv = report_dir / "unified_perf_summary.csv"
    if force or not unified_csv.exists():
        report_dir.mkdir(parents=True, exist_ok=True)
        from TraceLens.Reporting.generate_perf_report_pytorch import (  # noqa: PLC0415
            generate_perf_report_pytorch,
        )

        generate_perf_report_pytorch(
            profile_json_path=str(trace_path),
            output_xlsx_path=None,
            output_csvs_dir=str(report_dir),
            kernel_summary=True,
            include_first_occurrence_time=True,
            enable_pseudo_ops=enable_pseudo_ops,
        )
    return import_report_dir(store, report_dir, trace_path=trace_path, root=root)
