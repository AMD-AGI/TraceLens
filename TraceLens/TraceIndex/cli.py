###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Command line entry point for TraceIndex."""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from TraceLens.TraceIndex.importer import (
    append_trace,
    build_traces,
    report_dir_for_trace,
)
from TraceLens.TraceIndex.sqlite_store import SQLiteTraceIndexStore
from TraceLens.TraceIndex.utils import collect_trace_paths

DEFAULT_DB = Path("trace_index.sqlite")


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def create_store(args: argparse.Namespace):
    if args.backend != "sqlite":
        raise ValueError("only the sqlite backend is currently implemented")
    return SQLiteTraceIndexStore(args.db)


def append_cmd(args: argparse.Namespace) -> int:
    store = create_store(args)
    try:
        report_dir = args.report_dir
        generated = report_dir is None
        if generated:
            report_dir = report_dir_for_trace(args.trace_path, args.report_root)
        trace_id = append_trace(
            store,
            trace_path=args.trace_path,
            report_dir=None if generated else report_dir,
            root=args.root,
            force=args.force,
            enable_pseudo_ops=args.enable_pseudo_ops,
            report_root=args.report_root,
        )
        print_json(
            {
                "backend": args.backend,
                "db": args.db,
                "trace_id": trace_id,
                "trace_path": args.trace_path,
                "report_dir": report_dir,
                "generated_report": generated,
            }
        )
        return 0
    finally:
        store.close()


def build_cmd(args: argparse.Namespace) -> int:
    trace_paths = collect_trace_paths(args.traces_file, args.trace_path)
    if not trace_paths:
        raise SystemExit(
            "build requires --traces-file and/or one or more --trace-path values"
        )
    store = create_store(args)
    try:
        result = build_traces(
            store,
            trace_paths,
            report_root=args.report_root,
            root=args.root,
            force=args.force,
            enable_pseudo_ops=args.enable_pseudo_ops,
        )
        print_json(
            {
                "backend": args.backend,
                "db": args.db,
                "imported": result["imported"],
                "failed": result["failed"],
            }
        )
        return 1 if result["failed"] else 0
    finally:
        store.close()


def search_cmd(args: argparse.Namespace) -> int:
    store = create_store(args)
    try:
        store.init_schema()
        rows = [
            hit._asdict()
            for hit in store.search(" ".join(args.terms), limit=args.limit)
        ]
        print_json({"backend": args.backend, "rows": rows})
        return 0
    finally:
        store.close()


def sqlite_sql_cmd(args: argparse.Namespace) -> int:
    store = create_store(args)
    try:
        store.init_schema()
        rows = store.execute_read_query(args.sql, limit=args.limit)
        print_json({"backend": args.backend, "rows": rows})
        return 0
    finally:
        store.close()


def serve_cmd(args: argparse.Namespace) -> int:
    from TraceLens.TraceIndex.server import serve

    serve(
        db_path=args.db,
        host=args.host,
        port=args.port,
        default_limit=args.default_limit,
        max_limit=args.max_limit,
    )
    return 0


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Directory for generated CSV reports (default: trace_index_reports/)",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate CSV reports even if they already exist",
    )
    parser.add_argument("--enable-pseudo-ops", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query a TraceIndex catalog of traces."
    )
    parser.add_argument(
        "--backend",
        choices=["sqlite"],
        default="sqlite",
        help="TraceIndex storage backend",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB path")
    sub = parser.add_subparsers(dest="command")

    append = sub.add_parser(
        "append",
        help="Append one trace to the catalog, optionally from an existing CSV report",
    )
    append.add_argument("--trace-path", type=Path, required=True)
    append.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Existing TraceLens CSV report directory. If omitted, generate a training PyTorch report.",
    )
    add_generate_args(append)
    append.set_defaults(func=append_cmd)

    build = sub.add_parser(
        "build",
        help="Create or open the catalog and append a batch of traces",
    )
    build.add_argument(
        "--traces-file",
        type=Path,
        default=None,
        help="Text file with one trace path per line (# comments allowed)",
    )
    build.add_argument(
        "--trace-path",
        type=Path,
        action="append",
        default=[],
        help="Trace path to include. Repeatable, can be combined with --traces-file",
    )
    add_generate_args(build)
    build.set_defaults(func=build_cmd)

    search = sub.add_parser("search", help="Full-text search indexed traces")
    search.add_argument("terms", nargs="+")
    search.add_argument("--limit", type=int, default=50)
    search.set_defaults(func=search_cmd)

    sql = sub.add_parser("sqlite-sql", help="Run a read-only SQLite query")
    sql.add_argument("sql")
    sql.add_argument("--limit", type=int, default=500)
    sql.set_defaults(func=sqlite_sql_cmd)

    serve = sub.add_parser("serve", help="Serve read-only HTTP SQL access")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--default-limit", type=int, default=500)
    serve.add_argument("--max-limit", type=int, default=5000)
    serve.set_defaults(func=serve_cmd)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.error("a command is required")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
