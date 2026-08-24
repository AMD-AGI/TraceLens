###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend-neutral data objects for TraceIndex."""

from typing import Any, Dict, List, NamedTuple, Optional


class TraceRecord(NamedTuple):
    root: Optional[str]
    path: str
    rel_path: Optional[str]
    name: str
    size_bytes: Optional[int]
    md5: Optional[str]
    format: str
    rank: Optional[int]
    top_dir: Optional[str]
    parent_rel: Optional[str]
    should_enrich: bool
    skip_reason: Optional[str]


class TraceReport(NamedTuple):
    report_dir: str
    sheets: Dict[str, List[Dict[str, str]]]


class SearchHit(NamedTuple):
    trace_id: int
    rel_path: Optional[str]
    kind: str
    hit: str
