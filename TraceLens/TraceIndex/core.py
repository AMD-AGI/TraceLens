###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility facade for the default TraceIndex backend."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from TraceLens.TraceIndex.importer import (
    generate_report_and_import as generate_report_and_import_with_store,
)
from TraceLens.TraceIndex.importer import import_report_dir as import_report_dir_with_store
from TraceLens.TraceIndex.scanner import scan_traces as scan_traces_with_store
from TraceLens.TraceIndex.sqlite_store import SQLiteTraceIndexStore, is_read_only_sql


def scan_traces(
    db_path: Path,
    root: Path,
    peek_mb: int = 2,
    compute_md5: bool = False,
) -> int:
    store = SQLiteTraceIndexStore(db_path)
    try:
        return scan_traces_with_store(
            store,
            root=root,
            peek_mb=peek_mb,
            compute_md5=compute_md5,
        )
    finally:
        store.close()


def import_report_dir(
    db_path: Path,
    report_dir: Path,
    trace_path: Optional[Path] = None,
    root: Optional[Path] = None,
) -> int:
    store = SQLiteTraceIndexStore(db_path)
    try:
        return import_report_dir_with_store(
            store,
            report_dir=report_dir,
            trace_path=trace_path,
            root=root,
        )
    finally:
        store.close()


def generate_report_and_import(
    db_path: Path,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> int:
    store = SQLiteTraceIndexStore(db_path)
    try:
        return generate_report_and_import_with_store(
            store,
            trace_path=trace_path,
            report_dir=report_dir,
            root=root,
            force=force,
            enable_pseudo_ops=enable_pseudo_ops,
        )
    finally:
        store.close()


def search_index(db_path: Path, terms: str, limit: int = 50) -> List[Dict[str, Any]]:
    store = SQLiteTraceIndexStore(db_path)
    try:
        store.init_schema()
        return [hit._asdict() for hit in store.search(terms, limit=limit)]
    finally:
        store.close()


def execute_read_query(
    db_path: Path,
    sql: str,
    params: Optional[Sequence[Any]] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    store = SQLiteTraceIndexStore(db_path)
    try:
        store.init_schema()
        return store.execute_read_query(sql, params=params, limit=limit)
    finally:
        store.close()
