###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Typed schema for the grouped ``analysis.json`` artifact.

The contract between the ``render_analysis_json`` producer and any downstream
consumer of ``analysis.json``. Additive-only: a drifted field surfaces as a
``TypedDict`` mismatch at the read boundary, never a silent mis-parse. Carries
the compute, fusion and system tiers (standalone only).
``impact.mid`` is not additive across tiers: medium/low-confidence fusion
candidates do not null out their compute rows, so the same time can appear in
both ``compute_optimizations`` and ``fusion_optimizations``.
"""

from typing import TypedDict

try:  # NotRequired landed in typing in 3.11; earlier versions get it from typing_extensions.
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired


class Impact(TypedDict):
    """Task-level impact estimate (% E2E). No low/high band: a finding split
    across tasks has no per-task band, so per-finding bands live in the md only."""

    mid: float


class TaskMember(TypedDict):
    """One data-table row within a task (array-of-structs member)."""

    kernel_launcher_path: str | None
    library: str | None
    category: str | None
    analysis_md_rank: str | None
    kernel_name: list[str]
    args_shapes: list[str] | None
    args_datatypes: list[str] | None
    time_ms: float | None
    count: int | None
    pct_e2e: float | None
    flops_per_byte: float | None
    efficiency_percent: float | None
    efficiency_peak_value: float | None
    efficiency_peak_unit: str | None
    bound: str | None


class ComputeMember(TaskMember):
    """A compute task member; adds the per-row ``impact_score``."""

    impact_score: float


class _TaskBase(TypedDict):
    """Fields shared by all task tiers."""

    priority: int
    operation: str | None
    identification: str | None
    reasoning: str | None
    resolution: str | None
    prose_truncated: bool
    node: str | None


class ComputeTask(_TaskBase):
    """An operation-grouped compute optimization task."""

    impact: Impact
    members: list[ComputeMember]


class FusionTask(_TaskBase):
    """One kernel-fusion candidate; ``operation`` is the heading's pattern name.

    The severity color and ``P<N>:`` prefix are stripped from the heading.

    ``reasoning`` is always null (fusion blocks carry no such label) and
    ``priority`` is the rank within fusion by ``impact.mid``.
    """

    impact: Impact
    members: list[TaskMember]


class SystemMember(TypedDict):
    """One ``| Metric | Value |`` row of a system finding; ``value`` is null when unreadable."""

    metric: str
    value: float | None
    unit: str


class SystemTask(_TaskBase):
    """One system finding; ``operation`` is its fixed title.

    System impact is not quantifiable from trace data, so ``impact`` is always
    null, and ``priority`` is the report P-number.
    """

    impact: None
    members: list[SystemMember]


class ReportInfo(TypedDict):
    """Envelope rollup read entirely from report-level markers."""

    mode: str
    comparison_scope: str
    trace_quality: str
    warnings: NotRequired[list[str]]


class TopOperationRow(TypedDict):
    """One row of the flat Top Operations table."""

    rank: int
    category: str | None
    time_ms: float | None
    pct_of_compute: float | None
    ops: int | None
    potential_improvement_md: str | None
    node: NotRequired[str]


class ExecutiveSummary(TypedDict):
    """Narrative plus the wall-clock partition metrics."""

    narrative: str
    metrics: dict
    node_metrics: NotRequired[dict]


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
    system_optimizations: list[SystemTask]
    appendix: NotRequired[Appendix]
