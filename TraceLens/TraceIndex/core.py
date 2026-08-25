###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compatibility facade for the default TraceIndex backend."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from TraceLens.TraceIndex.importer import (
    append_trace as append_trace_with_store,
    build_traces as build_traces_with_store,
    generate_report_and_import as generate_report_and_import_with_store,
)
from TraceLens.TraceIndex.sqlite_store import SQLiteTraceIndexStore


def append_trace(
    db_path: Path,
    trace_path: Path,
    report_dir: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
    report_root: Optional[Path] = None,
) -> int:
    store = SQLiteTraceIndexStore(db_path)
    try:
        return append_trace_with_store(
            store,
            trace_path=trace_path,
            report_dir=report_dir,
            root=root,
            force=force,
            enable_pseudo_ops=enable_pseudo_ops,
            report_root=report_root,
        )
    finally:
        store.close()


def build_traces(
    db_path: Path,
    trace_paths: List[Path],
    report_root: Optional[Path] = None,
    root: Optional[Path] = None,
    force: bool = False,
    enable_pseudo_ops: bool = False,
) -> Dict[str, Any]:
    store = SQLiteTraceIndexStore(db_path)
    try:
        return build_traces_with_store(
            store,
            trace_paths,
            report_root=report_root,
            root=root,
            force=force,
            enable_pseudo_ops=enable_pseudo_ops,
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
