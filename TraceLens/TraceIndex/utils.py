###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared TraceIndex helpers that are independent of a storage backend."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def rel_to(path: Path, root: Path) -> str:
    try:
        return normalize_path(path.relative_to(root))
    except ValueError:
        return normalize_path(path)


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


def as_bool_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in (None, "", "nan", "NaN"):
        return 0
    text = str(value).strip().lower()
    return int(text in {"1", "true", "yes", "y"})


def search_text(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if part not in (None, "", "nan", "NaN"))
