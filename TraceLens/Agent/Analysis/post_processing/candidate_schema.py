###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Typed schema for the grouped ``analysis.json`` artifact.

The contract between the ``render_analysis_json`` producer and any downstream
consumer of ``analysis.json``. Additive-only: a drifted field surfaces as a
``TypedDict`` mismatch at the read boundary, never a silent mis-parse. Carries
the compute and fusion tiers (standalone only); there is no system task yet.
``impact.mid`` is not additive across tiers: medium/low-confidence fusion
candidates do not null out their compute rows, so the same time can appear in
both ``compute_optimizations`` and ``fusion_optimizations``.
"""

from typing import Optional, TypedDict

try:  # NotRequired landed in typing in 3.11; earlier versions get it from typing_extensions.
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class Impact(TypedDict):
    """Task-level impact estimate; ``mid`` is the point estimate."""

    mid: float
    low: Optional[float]
    high: Optional[float]


class TaskMember(TypedDict):
    """One data-table row within a task (array-of-structs member)."""

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


class ComputeMember(TaskMember):
    """A compute task member; adds the per-row ``impact_score``."""

    impact_score: float


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


class FusionTask(TypedDict):
    """One kernel-fusion candidate; ``operation`` is the heading's pattern name.

    The severity color and ``P<N>:`` prefix are stripped from the heading.

    ``reasoning`` is always null (fusion blocks carry no such label) and
    ``priority`` is the rank within fusion by ``impact.mid``.
    """

    priority: int
    operation: Optional[str]
    identification: Optional[str]
    reasoning: Optional[str]
    resolution: Optional[str]
    prose_truncated: bool
    impact: Impact
    members: list[TaskMember]


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
    fusion_optimizations: list[FusionTask]
    appendix: NotRequired[Appendix]
