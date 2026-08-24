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

from TraceLens.TraceIndex.core import (
    execute_read_query,
    generate_report_and_import,
    import_report_dir,
    scan_traces,
    search_index,
)


DEFAULT_DB = Path("trace_index.sqlite")


def print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def scan_cmd(args: argparse.Namespace) -> int:
    count = scan_traces(
        db_path=args.db,
        root=args.root,
        peek_mb=args.peek_mb,
        compute_md5=args.compute_md5,
    )
    print_json({"db": args.db, "root": args.root, "candidate_traces": count})
    return 0


def import_report_cmd(args: argparse.Namespace) -> int:
    trace_id = import_report_dir(
        db_path=args.db,
        report_dir=args.report_dir,
        trace_path=args.trace_path,
        root=args.root,
    )
    print_json({"db": args.db, "trace_id": trace_id, "report_dir": args.report_dir})
    return 0


def build_cmd(args: argparse.Namespace) -> int:
    trace_id = generate_report_and_import(
        db_path=args.db,
        trace_path=args.trace_path,
        report_dir=args.report_dir,
        root=args.root,
        force=args.force,
        enable_pseudo_ops=args.enable_pseudo_ops,
    )
    print_json({"db": args.db, "trace_id": trace_id, "trace_path": args.trace_path})
    return 0


def search_cmd(args: argparse.Namespace) -> int:
    rows = search_index(args.db, " ".join(args.terms), limit=args.limit)
    print_json({"rows": rows})
    return 0


def sql_cmd(args: argparse.Namespace) -> int:
    rows = execute_read_query(args.db, args.sql, limit=args.limit)
    print_json({"rows": rows})
    return 0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and query a SQLite index of TraceLens reports."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB path")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="Catalog trace-like files under a root")
    scan.add_argument("--root", type=Path, required=True)
    scan.add_argument("--peek-mb", type=int, default=2)
    scan.add_argument("--compute-md5", action="store_true")
    scan.set_defaults(func=scan_cmd)

    import_report = sub.add_parser(
        "import-report",
        help="Import an existing TraceLens CSV report directory",
    )
    import_report.add_argument("--report-dir", type=Path, required=True)
    import_report.add_argument("--trace-path", type=Path, default=None)
    import_report.add_argument("--root", type=Path, default=None)
    import_report.set_defaults(func=import_report_cmd)

    build = sub.add_parser(
        "build",
        help="Generate a TraceLens CSV report for one trace, then import it",
    )
    build.add_argument("--trace-path", type=Path, required=True)
    build.add_argument("--report-dir", type=Path, default=None)
    build.add_argument("--root", type=Path, default=None)
    build.add_argument("--force", action="store_true")
    build.add_argument("--enable-pseudo-ops", action="store_true")
    build.set_defaults(func=build_cmd)

    search = sub.add_parser("search", help="Full-text search indexed traces")
    search.add_argument("terms", nargs="+")
    search.add_argument("--limit", type=int, default=50)
    search.set_defaults(func=search_cmd)

    sql = sub.add_parser("sql", help="Run a read-only SQL query")
    sql.add_argument("sql")
    sql.add_argument("--limit", type=int, default=500)
    sql.set_defaults(func=sql_cmd)

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
