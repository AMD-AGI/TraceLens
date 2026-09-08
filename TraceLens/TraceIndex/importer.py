###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend-neutral TraceLens report import workflow."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from TraceLens.TraceIndex.models import HandoffEntry, TraceRecord, TraceReport
from TraceLens.TraceIndex.scanner import trace_record_from_path
from TraceLens.TraceIndex.store import TraceIndexStore
from TraceLens.TraceIndex.utils import normalize_path, read_csv_rows

DEFAULT_REPORT_ROOT = Path("trace_index_reports")

REPORT_SHEETS = (
    "unified_perf_summary",
    "ops_summary_by_category",
    "gpu_timeline",
)


def load_report_dir(
    report_dir: Path,
    excel_path: Optional[Path] = None,
) -> TraceReport:
    report_dir = report_dir.resolve()
    return TraceReport(
        report_dir=normalize_path(report_dir),
        sheets={
            sheet_name: read_csv_rows(report_dir / ("%s.csv" % sheet_name))
            for sheet_name in REPORT_SHEETS
        },
        excel_path=normalize_path(excel_path.resolve()) if excel_path else None,
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
    tracelens_id: Optional[str] = None,
    excel_path: Optional[Path] = None,
) -> int:
    store.init_schema()
    trace = (
        trace_record_from_path(trace_path, root=root)
        if trace_path is not None
        else synthetic_trace_record_for_report(report_dir)
    )
    if tracelens_id:
        trace = trace._replace(tracelens_id=tracelens_id)
    trace_id = store.upsert_trace(trace)
    store.import_report(trace_id, load_report_dir(report_dir, excel_path=excel_path))
    return trace_id


def load_handoff_jsonl(handoff_path: Path) -> List[HandoffEntry]:
    entries = []
    with handoff_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            required = ("tracelens_id", "trace_path", "report_path")
            missing = [key for key in required if not payload.get(key)]
            if missing:
                raise ValueError(
                    "handoff line %d is missing %s" % (line_number, ", ".join(missing))
                )
            entries.append(
                HandoffEntry(
                    tracelens_id=str(payload["tracelens_id"]),
                    trace_path=str(payload["trace_path"]),
                    report_path=str(payload["report_path"]),
                    excel_path=(
                        str(payload["excel_path"])
                        if payload.get("excel_path")
                        else None
                    ),
                )
            )
    return entries


def import_handoff_jsonl(
    store: TraceIndexStore,
    handoff_path: Path,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Import runner-produced artifact locations without walking runner trees."""
    imported: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for entry in load_handoff_jsonl(handoff_path):
        try:
            trace_id = import_report_dir(
                store,
                Path(entry.report_path),
                trace_path=Path(entry.trace_path),
                root=root,
                tracelens_id=entry.tracelens_id,
                excel_path=Path(entry.excel_path) if entry.excel_path else None,
            )
            imported.append(
                {
                    "tracelens_id": entry.tracelens_id,
                    "trace_id": trace_id,
                    "trace_path": entry.trace_path,
                    "report_path": entry.report_path,
                    "excel_path": entry.excel_path,
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "tracelens_id": entry.tracelens_id,
                    "trace_path": entry.trace_path,
                    "report_path": entry.report_path,
                    "error": repr(exc),
                }
            )
    return {"imported": imported, "failed": failed}


def report_dir_for_trace(trace_path: Path, report_root: Optional[Path] = None) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_path.name)
    return (report_root or DEFAULT_REPORT_ROOT) / safe_name


def append_trace(
    store: TraceIndexStore,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
    report_root: Optional[Path] = None,
) -> int:
    """Append one trace to the catalog.

    If ``report_dir`` is set, import that existing CSV report. Otherwise generate
    a training PyTorch CSV report and import it.
    """
    if report_dir is not None:
        return import_report_dir(store, report_dir, trace_path=trace_path, root=root)
    return generate_report_and_import(
        store,
        trace_path=trace_path,
        report_dir=report_dir_for_trace(trace_path, report_root),
        root=root,
        force=force,
        enable_pseudo_ops=enable_pseudo_ops,
    )


def build_traces(
    store: TraceIndexStore,
    trace_paths: List[Path],
    report_root: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> Dict[str, Any]:
    """Generate reports and append a batch of traces. Continues after failures."""
    imported: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for trace_path in trace_paths:
        report_dir = report_dir_for_trace(trace_path, report_root)
        try:
            trace_id = generate_report_and_import(
                store,
                trace_path=trace_path,
                report_dir=report_dir,
                root=root,
                force=force,
                enable_pseudo_ops=enable_pseudo_ops,
            )
            imported.append(
                {
                    "trace_id": trace_id,
                    "trace_path": normalize_path(trace_path),
                    "report_dir": normalize_path(report_dir),
                }
            )
        except Exception as exc:
            failed.append(
                {
                    "trace_path": normalize_path(trace_path),
                    "error": repr(exc),
                }
            )
    return {"imported": imported, "failed": failed}


def generate_report_and_import(
    store: TraceIndexStore,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> int:
    if report_dir is None:
        report_dir = report_dir_for_trace(trace_path)
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
