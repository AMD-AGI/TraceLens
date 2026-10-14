###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared test helpers migrated from test_agent_coverage.py."""

import os
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "TraceLens", "Agent", "Analysis")


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def _valid_compute_findings(rank=1, row=None):
    row = row or (
        "| aten::mm | M=2,N=3 | path/to/launch | mm_kernel | "
        "1.0 | 10 | 5 | 2000 | 50% | compute |"
    )
    header = (
        "| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | "
        "Count | FLOPS/Byte | Efficiency | Bound |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    return (
        "## Recommendations\n\n"
        f"### P{rank}: Optimize gemm\n"
        "**Insight**\n slow\n**Action**\n tune\n**Impact**\n high\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        f"<!-- reasoning-candidate tier=compute rank={rank} -->\n"
        "**Data:**\n\n"
        f"{header}\n{sep}\n{row}\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


def _valid_system_findings():
    return (
        "## Recommendations\n\n"
        "### P1: Fix idle\n"
        "**Insight**\n idle high\n**Action**\n overlap\n**Impact**\n medium\n"
        "<!-- impact-begin kind=p_item low=1.0 mid=2.0 high=3.0 -->\n"
        "<!-- impact-end -->\n\n"
        "## Detailed Analysis\n\n"
        "<!-- reasoning-candidate tier=system rank=1 -->\n"
        "System detail block.\n"
        "<!-- impact-begin kind=detail_estimate low=1.0 high=3.0 -->\n"
        "<!-- impact-end -->\n"
    )


def _full_report(extra_kf_impact=""):
    kf = "## Kernel Fusion Opportunities (Experimental)\n\n"
    if extra_kf_impact:
        kf += extra_kf_impact + "\n"
    return f"""# Analysis Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Time | 1000 ms |
| Compute % | 99.8% |
| Idle % | 0.2% |
| Exposed Communication % | 0.05% |
| Top Bottleneck Category | gemm |

## Compute Kernel Optimizations

### P1: Optimize gemm
<!-- impact-begin kind=p_item category=gemm low=85.0 mid=100.0 high=115.0 -->
Detail text for gemm optimization goes here with enough content.
<!-- impact-end -->

## {kf.split('## ')[1]}

## System-Level Optimizations

Top Operations:

<!-- impact-begin kind=top_ops -->

| Op | Time |
|----|------|
| gemm | 100 |

<!-- impact-end -->

## Detailed Analysis

### Compute Kernel Insights

#### P1: gemm detail
<!-- reasoning-candidate tier=compute rank=1 -->
More detail here.

### System-Level Insights

#### P1: idle detail
<!-- reasoning-candidate tier=system rank=1 -->
System detail.

## Appendix

Reference material.
"""


class _StubTree:
    def __init__(self, events, uid_map, parent_map=None):
        self.events = events
        self._uid_map = uid_map
        self._parent_map = parent_map or {}

    def get_UID2event(self, uid):
        return self._uid_map[uid]

    def get_parent_event(self, ev):
        return self._parent_map.get(id(ev))


class _StubAnalyzer:
    def __init__(self, tree, unified_events=None):
        self.tree = tree
        self._unified = unified_events or []

    def event_to_category(self, ev):
        return ev.get("_category", "aten")

    def collect_unified_perf_events(self):
        return self._unified


def _kernel_event(uid, name, dur=1000):
    return {"name": name, "dur": dur, "_category": "kernel", "gpu_events": []}


def _write_minimal_orchestrator_csvs(base, comparative=False):
    t1 = os.path.join(
        base, "perf_report_trace1_csvs" if comparative else "perf_report_csvs"
    )
    os.makedirs(t1)
    pd.DataFrame(
        {
            "type": ["total_time", "computation_time", "idle_time"],
            "time ms": [1000.0, 900.0, 100.0],
            "percent": [100.0, 90.0, 10.0],
        }
    ).to_csv(os.path.join(t1, "gpu_timeline.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "total_direct_kernel_time_ms": [800.0],
            "op category": ["GEMM"],
        }
    ).to_csv(os.path.join(t1, "ops_summary.csv"), index=False)
    pd.DataFrame(
        {
            "name": ["aten::mm"],
            "op category": ["GEMM"],
            "Kernel Time (µs)_sum": [800_000.0],
            "total_duration_us": [900_000.0],
        }
    ).to_csv(os.path.join(t1, "unified_perf_summary.csv"), index=False)
    if comparative:
        t2 = os.path.join(base, "perf_report_trace2_csvs")
        os.makedirs(t2)
        pd.DataFrame(
            {
                "type": ["total_time", "computation_time", "idle_time"],
                "time ms": [900.0, 810.0, 90.0],
                "percent": [100.0, 90.0, 10.0],
            }
        ).to_csv(os.path.join(t2, "gpu_timeline.csv"), index=False)
        pd.DataFrame(
            {
                "op category": ["GEMM"],
                "Kernel Time (µs)_sum": [700_000.0],
                "operation_count": [1],
            }
        ).to_csv(os.path.join(t2, "unified_perf_summary.csv"), index=False)
