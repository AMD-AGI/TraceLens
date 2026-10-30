###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pinned Model Explorer visualizer assets.

The component and its static files follow the
``ai-edge-model-explorer-visualizer`` README: the page loads the custom
element from unpkg, ``assetFilesBaseUrl`` may point at that package's
``dist/static_files``, and ``workerScriptPath`` must be a same-origin URL.
This module downloads the pinned ``dist/worker.js`` so the local server can
serve it. The bytes come from ``github._download_bytes``, the same opener
every other fetch uses. The Apache-2.0 license and attribution live next
to the viewer.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from TraceLens.ModelUtils.github import _download_bytes

VISUALIZER_VERSION = "0.1.2"
VISUALIZER_PACKAGE = "ai-edge-model-explorer-visualizer"
COMPONENT_SCRIPT_URL = f"https://unpkg.com/{VISUALIZER_PACKAGE}@{VISUALIZER_VERSION}"
STATIC_FILES_BASE_URL = f"{COMPONENT_SCRIPT_URL}/dist/static_files"
WORKER_URL = f"https://cdn.jsdelivr.net/npm/{VISUALIZER_PACKAGE}@{VISUALIZER_VERSION}/dist/worker.js"
# dist/worker.js from ai-edge-model-explorer-visualizer 0.1.2.
WORKER_SHA256 = "f857968b63b0a40fb48ed2d63d50e86a73be943c558f4098fd44855735efd738"

_CACHE_ROOT = Path.home() / ".cache" / "tracelens" / "model-explorer"
_CACHE_LOCK = threading.Lock()


def worker_cache_path() -> Path:
    return _CACHE_ROOT / VISUALIZER_VERSION / "worker.js"


def worker_js_bytes() -> bytes:
    """Return the pinned layout worker, downloading it on a cache miss."""
    path = worker_cache_path()
    cached = _cached_worker(path)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _cached_worker(path)
        if cached is not None:
            return cached
        data = _download_bytes(WORKER_URL)
        digest = hashlib.sha256(data).hexdigest()
        if digest != WORKER_SHA256:
            raise RuntimeError(
                f"{VISUALIZER_PACKAGE}@{VISUALIZER_VERSION} worker.js sha256 {digest} "
                f"does not match {WORKER_SHA256}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return data


def _cached_worker(path: Path) -> bytes | None:
    if not path.is_file():
        return None
    cached = path.read_bytes()
    if hashlib.sha256(cached).hexdigest() != WORKER_SHA256:
        return None
    return cached
