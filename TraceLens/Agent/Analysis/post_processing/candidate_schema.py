###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Typed schema for the grouped ``analysis.json`` artifact.

The contract between the ``render_analysis_json`` producer and any downstream
consumer of ``analysis.json``. Additive-only: a drifted field surfaces as a
``TypedDict`` mismatch at the read boundary, never a silent mis-parse. Compute
tier only; no fusion/system task currnetly``.
"""

from typing import NotRequired, Optional, TypedDict


class Impact(TypedDict):
    """Task-level impact estimate; ``mid`` is the point estimate."""

    mid: float
    low: Optional[float]
    high: Optional[float]


class ComputeMember(TypedDict):
    """One data-table row within a compute task (array-of-structs member)."""

    impact_score: float
    kernel_launcher_path: Optional[str]
    library: Optional[str]
    category: Optional[str]
    analysis_md_rank: Optional[str]
    kernel_name: list[str]
    args_shapes: Optional[list[str]]
    args_datatypes: Optional[list[str]]
    time_ms: Optional[float]
    count: Optional[int]
    pct_e2e: Optional[float]
    flops_per_byte: Optional[float]
    efficiency_percent: Optional[float]
    efficiency_peak_value: Optional[float]
    efficiency_peak_unit: Optional[str]
    bound: Optional[str]


class ComputeTask(TypedDict):
    """An operation-grouped compute optimization task."""

    priority: int
    operation: Optional[str]
    identification: Optional[str]
    reasoning: Optional[str]
    resolution: Optional[str]
    prose_truncated: bool
    impact: Impact
    members: list[ComputeMember]


class ReportInfo(TypedDict):
    """Envelope rollup read entirely from report-level markers."""

    mode: str
    comparison_scope: str
    trace_quality: str
    warnings: NotRequired[list[str]]


class TopOperationRow(TypedDict):
    """One row of the flat Top Operations table."""

    rank: int
    category: Optional[str]
    time_ms: Optional[float]
    pct_of_compute: Optional[float]
    ops: Optional[int]
    potential_improvement_md: Optional[str]


class ExecutiveSummary(TypedDict):
    """Narrative plus the wall-clock partition metrics."""

    narrative: str
    metrics: dict


class Appendix(TypedDict):
    """Model architecture and hardware reference blocks."""

    model_architecture: dict
    hardware_reference: dict


class AnalysisReport(TypedDict):
    """Top-level ``analysis.json`` envelope."""

    report_info: ReportInfo
    title: str
    executive_summary: NotRequired[ExecutiveSummary]
    top_operations: NotRequired[list[TopOperationRow]]
    compute_optimizations: list[ComputeTask]
    appendix: NotRequired[Appendix]
