"""MCP application — TraceLens GPU performance analysis server.

Exposes 3 core tools:
  1. check_trace_file — verify a trace file path before analysis
  2. run_full_standalone_analysis — one-call complete analysis pipeline
  3. run_comparative_analysis — compare two GPU traces
"""

import json
import logging
import os
from enum import Enum
from typing import Optional

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import standalone_tools, comparative_tools
from .config import config
from .standalone_tools import _cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "tracelens",
    instructions=(
        "TraceLens GPU performance analysis server. "
        "Trace files are on the shared NFS — provide the absolute path. "
        "Use `check_trace_file` to verify, then `run_full_standalone_analysis` "
        "to get all metrics in one call. Use the returned data to write "
        "a standalone_analysis.md report."
    ),
    host=config.host,
    port=config.port,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Platform(str, Enum):
    MI300X = "MI300X"
    MI325X = "MI325X"
    MI350X = "MI350X"
    MI355X = "MI355X"
    MI400 = "MI400"


class TraceType(str, Enum):
    pytorch = "pytorch"
    jax = "jax"
    rocprof = "rocprof"


# ---------------------------------------------------------------------------
# Tool 1: check_trace_file
# ---------------------------------------------------------------------------

@mcp.tool()
def check_trace_file(trace_path: str) -> dict:
    """Check if a trace file exists on the server and return its metadata.
    Use this to verify a path before running analysis.
    """
    p = os.path.realpath(trace_path)
    if not os.path.exists(p):
        return {"exists": False, "path": p, "error": "File not found."}
    if not os.path.isfile(p):
        return {"exists": False, "path": p, "error": "Path is not a file."}

    stat = os.stat(p)
    size_mb = round(stat.st_size / (1024 ** 2), 2)

    if p.endswith(".json.gz"):
        file_type = "json.gz (compressed trace)"
    elif p.endswith(".json"):
        file_type = "json (trace)"
    elif p.endswith(".xlsx"):
        file_type = "xlsx (perf report)"
    elif p.endswith(".pb"):
        file_type = "pb (JAX XPlane)"
    else:
        file_type = f"unknown ({os.path.splitext(p)[-1]})"

    return {"exists": True, "path": p, "size_mb": size_mb, "file_type": file_type}


# ---------------------------------------------------------------------------
# Tool 2: run_full_standalone_analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def run_full_standalone_analysis(
    trace_path: str,
    platform: Platform,
    output_dir: Optional[str] = None,
    trace_type: TraceType = TraceType.pytorch,
    enable_pseudo_ops: bool = True,
    cleanup: bool = True,
) -> dict:
    """Run the COMPLETE standalone analysis pipeline in one call.

    Give it a trace file path (absolute NFS path) and GPU platform.
    It runs every step automatically and returns all metrics.

    When cleanup=True (default), intermediate files are deleted after
    metrics are collected — zero disk footprint.

    Use the returned data to write the final standalone_analysis.md report.
    """
    return standalone_tools.run_full_standalone_analysis(
        trace_path=trace_path,
        platform=platform.value,
        output_dir=output_dir,
        trace_type=trace_type.value,
        enable_pseudo_ops=enable_pseudo_ops,
        cleanup=cleanup,
    )


# ---------------------------------------------------------------------------
# Tool 3: run_comparative_analysis
# ---------------------------------------------------------------------------

@mcp.tool()
def run_comparative_analysis(
    gpu1_kineto: str,
    gpu2_kineto: str,
    gpu1_name: Optional[str] = None,
    gpu2_name: Optional[str] = None,
    cleanup: bool = True,
) -> dict:
    """Compare two GPU traces (deterministic analysis, no LLM).
    
    When cleanup=True (default), intermediate files are deleted after
    analysis — zero disk footprint.
    
    Returns comparison data that can be used to write a comparison report.
    """
    return comparative_tools.run_comparative_analysis(
        gpu1_kineto=gpu1_kineto,
        gpu2_kineto=gpu2_kineto,
        gpu1_name=gpu1_name,
        gpu2_name=gpu2_name,
        cleanup=cleanup,
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("tracelens://platform-specs")
def get_platform_specs() -> str:
    """GPU platform specifications (HBM bandwidth, peak TFLOPS) for all supported platforms."""
    specs, _ = standalone_tools._get_platform_specs()
    return json.dumps(specs, indent=2)


# ---------------------------------------------------------------------------
# Prompt — standalone analysis workflow
# ---------------------------------------------------------------------------

@mcp.prompt()
def standalone_analysis(
    trace_path: str = "<trace_path>",
    platform: str = "<platform>",
) -> str:
    """Full standalone GPU performance analysis workflow."""
    return f"""Perform a full standalone GPU performance analysis using TraceLens.

## Steps

1. Call `check_trace_file` with trace_path="{trace_path}" to verify the file exists.

2. Call `run_full_standalone_analysis` with trace_path="{trace_path}", platform="{platform}".
   This runs the entire pipeline and returns all metrics in one call.

3. **IMPORTANT: Create a file** named `standalone_analysis.md` using the Write tool.
   The report MUST include these sections:
   - **Executive Summary** — GPU utilization, top 3-5 bottlenecks table, potential speedup
   - **System-Level Optimizations** — GPU idle time, compute/comm overlap, memcpy issues
   - **Compute Kernel Optimizations** — per-category analysis prioritized by gpu_kernel_time_ms
   - **Impact Summary** — table: Recommendation | Type | Estimated Savings (ms) | Confidence
   - **Appendix: Hardware Specs** — MUST include platform name, HBM bandwidth (GB/s), memory (GB), peak TFLOPS for BF16/FP16/FP8/FP32 (from report_instructions.platform_specs)

## Language Guidelines
Use vendor-agnostic terminology: GPU kernels, collective communication, vendor GEMM library.
Exception: when quoting kernel names from traces, include the actual name.

## Key Rules
- Use gpu_kernel_time_ms for bottleneck ranking — NOT CPU duration
- Flag any efficiency > 100% as [ANOMALY]
- If CPU duration >> GPU kernel time (>5x), flag as sync bottleneck
- Priority icons: P1=red (highest impact), P2=yellow, P3=green
- The Appendix MUST contain the full hardware specifications table
"""


# ---------------------------------------------------------------------------
# Custom HTTP routes
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "cached_traces": len(_cache._entries),
    })
