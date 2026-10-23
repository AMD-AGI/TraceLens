###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trace metadata extraction for catalog ingest."""

import gzip
import hashlib
import re
from pathlib import Path
from typing import Optional

from TraceLens.TraceIndex.models import TraceRecord
from TraceLens.TraceIndex.utils import classify_skip, normalize_path, rel_to

RANK_RE = re.compile(r"(?:^|[^A-Za-z])rank[-_]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)


def is_json_gz(path: Path) -> bool:
    return path.name.lower().endswith(".json.gz")


def read_prefix(path: Path, max_bytes: int) -> str:
    try:
        if is_json_gz(path):
            with gzip.open(path, "rb") as f:
                data = f.read(max_bytes)
        else:
            with path.open("rb") as f:
                data = f.read(max_bytes)
    except (OSError, EOFError, gzip.BadGzipFile):
        return ""
    return data.decode("utf-8", errors="ignore")


def content_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Optional[str]:
    hasher = hashlib.md5()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def detect_format(path: Path, prefix: str) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".pftrace":
        return "pftrace"
    if suffix == ".rpd":
        return "rocprof_rpd"
    if name.endswith(".xplane.pb"):
        return "xplane_pb"
    if '"traceEvents"' in prefix:
        return "kineto_chrome_json_gz" if is_json_gz(path) else "kineto_chrome_json"
    if '"rocprofiler-sdk-tool"' in prefix:
        return "rocprofv3_json_gz" if is_json_gz(path) else "rocprofv3_json"
    if is_json_gz(path):
        return "json_gz_unknown"
    if suffix == ".json":
        return "json_unknown"
    return "trace_named_unknown"


def extract_rank(path: Path) -> Optional[int]:
    match = RANK_RE.search(normalize_path(path))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def trace_record_from_path(
    trace_path: Path,
    root: Optional[Path] = None,
    peek_bytes: int = 2 * 1024 * 1024,
    compute_md5: bool = False,
) -> TraceRecord:
    trace_path = trace_path.resolve()
    root = root.resolve() if root is not None else trace_path.parent.resolve()
    prefix = read_prefix(trace_path, peek_bytes)
    trace_format = detect_format(trace_path, prefix)
    skip_reason = classify_skip(trace_path, root)
    should_enrich = skip_reason is None and trace_format in {
        "kineto_chrome_json",
        "kineto_chrome_json_gz",
        "rocprofv3_json",
        "rocprofv3_json_gz",
        "pftrace",
    }
    rel_path = rel_to(trace_path, root)
    rel_parts = Path(rel_path).parts
    return TraceRecord(
        root=normalize_path(root),
        path=normalize_path(trace_path),
        rel_path=rel_path,
        name=trace_path.name,
        size_bytes=trace_path.stat().st_size if trace_path.exists() else None,
        md5=content_md5(trace_path) if compute_md5 else None,
        format=trace_format,
        rank=extract_rank(trace_path),
        top_dir=rel_parts[0] if rel_parts else "",
        parent_rel=normalize_path(Path(*rel_parts[:-1])) if len(rel_parts) > 1 else "",
        should_enrich=should_enrich,
        skip_reason=skip_reason,
    )
