"""Compose TraceLens Model Explorer viewer HTML pages."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
PACKAGE_ROOT = Path(__file__).resolve().parent
VISUALIZER_DIST = PACKAGE_ROOT / "node_modules" / "ai-edge-model-explorer-visualizer" / "dist"
APP_JS_PATTERN = re.compile(r'    <script src="\./app\.js\?v=\d+"></script>')


def is_html_output(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".html"


def render_payload_script(payload: dict[str, Any]) -> str:
    """Embed export JSON safely inside HTML."""
    blob = json.dumps(payload, ensure_ascii=False)
    blob = blob.replace("</", "<\\/")
    return f'    <script id="tracelens-payload" type="application/json">{blob}</script>'


def _worker_js_source() -> Path:
    for candidate in (VISUALIZER_DIST / "worker.js", VIEWER_DIR / "worker.js"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "worker.js not found; install ai-edge-model-explorer-visualizer or bundle viewer/worker.js"
    )


def render_worker_script(worker_js: str) -> str:
    """Embed layout worker source for file:// exports (Workers cannot load local paths)."""
    safe = worker_js.replace("</", "<\\/")
    return f'    <script id="tracelens-worker-source" type="text/plain">{safe}</script>'


def compose_viewer_html(
    payload: dict[str, Any] | None = None,
    *,
    inline_app: bool = False,
) -> str:
    """Build viewer HTML, optionally embedding payload and app.js."""
    shell = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    app_js = (VIEWER_DIR / "app.js").read_text(encoding="utf-8")

    replacement_parts: list[str] = []
    if payload is not None:
        replacement_parts.append(render_payload_script(payload))
    if inline_app:
        replacement_parts.append(render_worker_script(_worker_js_source().read_text(encoding="utf-8")))
        replacement_parts.append(f"    <script>\n{app_js}\n    </script>")
    else:
        replacement_parts.append('    <script src="./app.js?v=10"></script>')

    if replacement_parts:
        replacement = "\n".join(replacement_parts)
        shell, count = APP_JS_PATTERN.subn(lambda _match: replacement, shell, count=1)
        if count != 1:
            raise RuntimeError("Viewer shell is missing the app.js script tag.")

    return shell


def save_viewer_html(payload: dict[str, Any], path: Path | str) -> Path:
    """Write a self-contained standalone viewer page (payload, worker, and app inline)."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(compose_viewer_html(payload, inline_app=True), encoding="utf-8")
    return target
