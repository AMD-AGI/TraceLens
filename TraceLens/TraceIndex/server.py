###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Read-only HTTP query server for the SQLite TraceIndex backend."""

import json
import sqlite3
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Type
from urllib.parse import parse_qs, urlparse

from TraceLens.TraceIndex.sqlite_store import is_read_only_sql


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def make_handler(db_path: Path, default_limit: int, max_limit: int) -> Type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "TraceIndexQuery/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

        def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def connect(self) -> sqlite3.Connection:
            uri = "file:%s?mode=ro" % db_path
            conn = sqlite3.connect(uri, uri=True, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            return conn

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_json(
                    {
                        "endpoints": {
                            "GET /health": "server and DB status",
                            "GET /tables": "table row counts",
                            "POST /query": {"sql": "SELECT ...", "params": [], "limit": 500},
                        }
                    }
                )
                return
            if parsed.path == "/health":
                self.handle_health()
                return
            if parsed.path == "/tables":
                self.handle_tables()
                return
            if parsed.path == "/query":
                params = parse_qs(parsed.query)
                sql = params.get("sql", [""])[0]
                limit = int(params.get("limit", [str(default_limit)])[0])
                self.handle_query(sql, [], limit)
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/query":
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self.read_json()
                self.handle_query(
                    str(payload.get("sql", "")),
                    payload.get("params", []),
                    int(payload.get("limit", default_limit)),
                )
            except Exception as exc:
                self.send_json({"error": repr(exc)}, HTTPStatus.BAD_REQUEST)

        def handle_health(self) -> None:
            self.send_json(
                {
                    "ok": db_path.exists(),
                    "db_path": str(db_path),
                    "db_size_mb": round(db_path.stat().st_size / 1048576, 2) if db_path.exists() else None,
                }
            )

        def handle_tables(self) -> None:
            with self.connect() as conn:
                tables = [
                    row["name"]
                    for row in conn.execute(
                        """
                        SELECT name
                        FROM sqlite_master
                        WHERE type IN ('table', 'view')
                          AND name NOT LIKE 'sqlite_%'
                          AND name NOT LIKE '%_data'
                          AND name NOT LIKE '%_idx'
                          AND name NOT LIKE '%_docsize'
                          AND name NOT LIKE '%_config'
                        ORDER BY name
                        """
                    )
                ]
                counts = {}
                for table in tables:
                    try:
                        counts[table] = conn.execute('SELECT COUNT(*) AS n FROM "%s"' % table).fetchone()["n"]
                    except sqlite3.DatabaseError as exc:
                        counts[table] = repr(exc)
                self.send_json({"tables": counts})

        def handle_query(self, sql: str, params: List[Any], limit: int) -> None:
            if not is_read_only_sql(sql):
                self.send_json({"error": "only single read-only SELECT/WITH/PRAGMA statements are allowed"}, HTTPStatus.BAD_REQUEST)
                return
            limit = max(1, min(limit, max_limit))
            start = time.perf_counter()
            with self.connect() as conn:
                rows = conn.execute(sql, params).fetchmany(limit + 1)
                elapsed_ms = (time.perf_counter() - start) * 1000
                returned = rows[:limit]
                self.send_json(
                    {
                        "elapsed_ms": round(elapsed_ms, 3),
                        "limit": limit,
                        "truncated": len(rows) > limit,
                        "rows": [dict(row) for row in returned],
                    }
                )

    return Handler


def serve(db_path: Path, host: str = "127.0.0.1", port: int = 8765, default_limit: int = 500, max_limit: int = 5000) -> None:
    handler = make_handler(db_path, default_limit, max_limit)
    server = ThreadingHTTPServer((host, port), handler)
    print("serving db=%s on http://%s:%s" % (db_path, host, port), flush=True)
    server.serve_forever()
