###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared TraceIndex helpers that are independent of a storage backend."""

import ast
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel_to(path: Path, root: Path) -> str:
    try:
        return normalize_path(path.relative_to(root))
    except ValueError:
        return normalize_path(path)


def _set_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)


_set_csv_field_size_limit()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def first_value(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    lower_map = {key.lower(): key for key in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is None:
            continue
        value = row.get(key)
        if value not in (None, "", "nan", "NaN"):
            return value
    return default


def as_text(value: Any) -> Optional[str]:
    if value in (None, "", "nan", "NaN"):
        return None
    return str(value)


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    number = as_float(value)
    return int(number) if number is not None else None


def as_duration_us(
    row: Dict[str, Any],
    us_names: Sequence[str],
    ms_names: Sequence[str] = (),
) -> Optional[float]:
    us_value = as_float(first_value(row, us_names))
    if us_value is not None:
        return us_value
    ms_value = as_float(first_value(row, ms_names))
    if ms_value is not None:
        return ms_value * 1000.0
    return None


def as_bool_int(value: Any) -> int:
    optional = as_optional_bool_int(value)
    return 0 if optional is None else optional


def as_optional_bool_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if value in (None, "", "nan", "NaN"):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def search_text(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if part not in (None, "", "nan", "NaN"))


NP_SCALAR_RE = re.compile(r"\b(?:np|numpy)\.(?:float|int)(?:16|32|64)?\(([^()]+)\)")


def clean_python_repr(text: str) -> str:
    cleaned = NP_SCALAR_RE.sub(r"\1", text)
    return cleaned.replace("nan", "None")


def parse_repr(text: Any) -> Any:
    if not text:
        return None
    if not isinstance(text, str):
        return text
    try:
        return ast.literal_eval(clean_python_repr(text))
    except (SyntaxError, ValueError, MemoryError):
        return None


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass
    return str(value)


def to_json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(json_safe(value), sort_keys=True)


def kernel_flags(name: str) -> Tuple[Optional[str], int, int, int]:
    low = name.lower()
    is_tensile = int("cijk" in low or "tensile" in low)
    is_transpose = int("transpose" in low or "permute" in low)
    is_layout = int(
        is_transpose
        or "contiguous" in low
        or "copy" in low
        or "cast" in low
        or "convert" in low
    )
    library = None
    if is_tensile:
        library = "Tensile"
    elif "triton" in low:
        library = "Triton"
    elif "ck" in low or "composable" in low:
        library = "CK"
    elif "nccl" in low or "rccl" in low:
        library = "RCCL/NCCL"
    return library, is_tensile, is_transpose, is_layout


SKIP_PARTS_EXACT = {
    ".git",
    "__pycache__",
    "node_modules",
    "_perf_report_csvs",
    "perf_report_csvs",
    "gap_analysis",
    "capture_traces",
    "graph_capture",
}
SKIP_PARTS_CONTAINS = (
    "_perf_report_csvs",
    "perf_report",
    "gap_analysis",
    "capture_traces",
    "graph_capture",
)
TRACE_FILE_SUFFIXES = (".json.gz", ".json", ".pftrace", ".rpd", ".xplane.pb")


def classify_skip(path: Path, root: Path) -> Optional[str]:
    try:
        rel_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
    except ValueError:
        rel_parts = []
    for part in rel_parts:
        if part in SKIP_PARTS_EXACT:
            return part
        for token in SKIP_PARTS_CONTAINS:
            if token in part:
                return token
    name = path.name.lower()
    if name.endswith((".xlsx", ".csv", ".log", ".jsonl", ".md", ".txt")):
        return "derived_or_log"
    return None


def is_trace_filename(path: Path) -> bool:
    return path.name.lower().endswith(TRACE_FILE_SUFFIXES)


def collect_traces_from_dir(trace_dir: Path) -> List[Path]:
    """Walk ``trace_dir`` for trace-like files, skipping report CSV dirs."""
    root = trace_dir.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    found: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not is_trace_filename(path):
            continue
        if classify_skip(path, root) is not None:
            continue
        found.append(path)
    found.sort(key=lambda item: normalize_path(item))
    return found


def read_traces_file(path: Path) -> List[Path]:
    """Read one trace path per line. Blank lines and ``#`` comments are ignored."""
    traces: List[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        traces.append(Path(line).expanduser())
    return traces


def _coerce_paths(
    value: Optional[Union[Path, Sequence[Path]]],
) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def collect_trace_paths(
    traces_file: Optional[Path] = None,
    trace_paths: Optional[Sequence[Path]] = None,
    trace_dirs: Optional[Sequence[Path]] = None,
) -> List[Path]:
    paths: List[Path] = []
    if traces_file is not None:
        paths.extend(read_traces_file(traces_file))
    paths.extend(_coerce_paths(trace_paths))
    for trace_dir in _coerce_paths(trace_dirs):
        paths.extend(collect_traces_from_dir(trace_dir))
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = normalize_path(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique
