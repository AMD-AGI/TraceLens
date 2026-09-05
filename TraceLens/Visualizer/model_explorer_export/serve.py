#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Serve the bundled Model Explorer viewer for exported TraceLens graphs."""

from __future__ import annotations

import json
import shutil
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from TraceLens.Visualizer.model_explorer_export.viewer_page import VIEWER_DIR, compose_viewer_html

PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALIZER_DIST = (
    PACKAGE_ROOT / "node_modules" / "ai-edge-model-explorer-visualizer" / "dist"
)


def ensure_viewer_assets() -> None:
    """Copy worker.js from the npm package when available."""
    worker_src = VISUALIZER_DIST / "worker.js"
    worker_dst = VIEWER_DIR / "worker.js"
    if worker_src.exists():
        worker_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(worker_src, worker_dst)


def _load_payload(
    *,
    payload: dict[str, Any] | None,
    json_path: Path | None,
) -> dict[str, Any]:
    if payload is not None:
        return payload
    if json_path is None:
        raise ValueError("Provide payload or json_path.")
    return json.loads(json_path.expanduser().resolve().read_text(encoding="utf-8"))


def viewer_url(port: int = 8765) -> str:
    """Return the local viewer URL for a given port."""
    return f"http://127.0.0.1:{port}/"


def serve_viewer(
    *,
    payload: dict[str, Any] | None = None,
    json_path: Path | None = None,
    port: int = 8765,
    block: bool = False,
) -> str:
    """Start a local HTTP server with the graph payload embedded in index.html."""
    ensure_viewer_assets()
    resolved_payload = _load_payload(payload=payload, json_path=json_path)
    served_index = compose_viewer_html(resolved_payload, inline_app=False)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path in {"", "/", "/index.html"}:
                body = served_index.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def end_headers(self) -> None:
            path = self.path.split("?", 1)[0]
            if path.endswith((".html", ".js", ".json")):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = viewer_url(port)

    def run() -> None:
        try:
            server.serve_forever()
        finally:
            server.server_close()

    thread = threading.Thread(target=run, daemon=not block)
    thread.start()

    if block:
        thread.join()

    return url


def open_viewer(url: str) -> None:
    webbrowser.open(url)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve a TraceLens Model Explorer viewer."
    )
    parser.add_argument(
        "json",
        type=Path,
        help="Model Explorer JSON exported by visualize_model_in_explorer.py",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    json_path = args.json.expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"File not found: {json_path}")

    url = viewer_url(args.port)
    print(f"Open viewer: {url}")
    if args.open:
        open_viewer(url)
    serve_viewer(json_path=json_path, port=args.port, block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
