#!/usr/bin/env python3
"""Serve the bundled Model Explorer viewer for exported TraceLens graphs."""

from __future__ import annotations

import shutil
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALIZER_DIST = PACKAGE_ROOT / "node_modules" / "ai-edge-model-explorer-visualizer" / "dist"


def ensure_viewer_assets() -> None:
    """Copy worker.js from the npm package when available."""
    worker_src = VISUALIZER_DIST / "worker.js"
    worker_dst = VIEWER_DIR / "worker.js"
    if worker_src.exists():
        worker_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(worker_src, worker_dst)


def serve_viewer(
    *,
    json_path: Path,
    port: int = 8765,
    block: bool = False,
) -> str:
    """Start a local HTTP server rooted at the viewer directory."""
    ensure_viewer_assets()
    json_path = json_path.expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    payload_name = json_path.name
    viewer_payload = VIEWER_DIR / payload_name
    shutil.copy2(json_path, viewer_payload)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIEWER_DIR), **kwargs)

        def end_headers(self) -> None:
            path = self.path.split("?", 1)[0]
            if path.endswith((".html", ".js", ".json")):
                self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/index.html?graph={quote(payload_name)}"

    def run() -> None:
        try:
            server.serve_forever()
        finally:
            server.server_close()
            viewer_payload.unlink(missing_ok=True)

    thread = threading.Thread(target=run, daemon=not block)
    thread.start()

    if block:
        thread.join()

    return url


def open_viewer(url: str) -> None:
    webbrowser.open(url)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve a TraceLens Model Explorer viewer.")
    parser.add_argument("json", type=Path, help="Model Explorer JSON exported by visualize_model_in_explorer.py")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    url = serve_viewer(json_path=args.json, port=args.port, block=True)
    print(url)
    if args.open:
        open_viewer(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
