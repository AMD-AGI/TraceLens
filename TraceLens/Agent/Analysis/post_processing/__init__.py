###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-processing utilities: the grouped ``analysis.json`` producer.

Sits after ``analysis.md`` generation and renders the structured, typed JSON
artifact downstream consumers read instead of scraping markdown. Re-exports the
public surface: the renderer and the ``analysis.json`` TypedDicts.
"""

from TraceLens.Agent.Analysis.post_processing.analysis_json import (
    render_analysis_json,
)
from TraceLens.Agent.Analysis.post_processing.candidate_schema import (
    AnalysisReport,
    Appendix,
    ComputeMember,
    ComputeTask,
    ExecutiveSummary,
    Impact,
    ReportInfo,
    TopOperationRow,
)

__all__ = [
    "render_analysis_json",
    "AnalysisReport",
    "Appendix",
    "ComputeMember",
    "ComputeTask",
    "ExecutiveSummary",
    "Impact",
    "ReportInfo",
    "TopOperationRow",
]
