"""Standalone analysis tools — run_full_standalone_analysis and its internal helpers.

Adapted from PR #23 MCPServer/standalone_tools.py.
Path resolution uses the same relative layout (MCPServer/ sits next to Standalone/).
"""

import json
import os
import sys
from typing import Optional

from .state import TraceCache

_cache = TraceCache()


def _get_standalone_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "Standalone")


def _get_platform_specs():
    standalone_dir = _get_standalone_dir()
    if standalone_dir not in sys.path:
        sys.path.insert(0, standalone_dir)
    from utils.platform_specs import PLATFORM_SPECS, CATEGORY_SKILL_MAP
    return PLATFORM_SPECS, CATEGORY_SKILL_MAP


# ---------------------------------------------------------------------------
# Internal pipeline steps (not exposed as MCP tools)
# ---------------------------------------------------------------------------

def _generate_perf_report(trace_path: str, output_dir: str,
                          trace_type: str = "pytorch",
                          enable_pseudo_ops: bool = True) -> dict:
    import subprocess

    csv_dir = os.path.join(output_dir, "perf_report_csvs")

    if trace_type == "jax":
        cmd = [
            sys.executable, "-m",
            "TraceLens.Reporting.generate_perf_report_jax",
            "--profile_pb_path", trace_path,
            "--output_csvs_dir", csv_dir,
        ]
    else:
        cmd = [
            sys.executable, "-m",
            "TraceLens.Reporting.generate_perf_report_pytorch",
            "--profile_json_path", trace_path,
            "--output_csvs_dir", csv_dir,
        ]
        if enable_pseudo_ops:
            cmd.append("--enable_pseudo_ops")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": f"Perf report generation failed (exit {result.returncode})",
                "stderr": result.stderr[-2000:] if result.stderr else ""}

    csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")] if os.path.isdir(csv_dir) else []
    return {"csv_dir": csv_dir, "csvs": csvs}


def _prepare_agentic(platform: str, output_dir: str,
                     trace_path: str = None,
                     enable_pseudo_ops: bool = True) -> dict:
    import subprocess

    script_path = os.path.join(_get_standalone_dir(), "orchestrator_prepare.py")
    effective_trace_path = trace_path or "none"

    cmd = [
        sys.executable, script_path,
        "--trace-path", effective_trace_path,
        "--platform", platform,
        "--output-dir", output_dir,
    ]
    if not enable_pseudo_ops:
        cmd.append("--disable_pseudo_ops")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": f"orchestrator_prepare failed (exit {result.returncode})",
                "stderr": result.stderr[-2000:] if result.stderr else ""}

    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return json.load(f)

    return {"error": "Manifest not generated. Check that perf_report_csvs/ exists."}


def _run_single_category(output_dir: str, category: str) -> dict:
    import subprocess

    module_map = {
        "gemm": "gemm_analysis",
        "sdpa_fwd": "sdpa_analysis",
        "moe_fused": "moe_analysis",
        "elementwise": "elementwise_analysis",
        "reduce": "reduce_analysis",
        "triton": "triton_analysis",
        "norm": "norm_analysis",
        "convolution": "convolution_analysis",
        "cpu_idle": "cpu_idle_analysis",
        "multi_kernel": "multi_kernel_analysis",
        "other": "other_analysis",
    }

    module_name = module_map.get(category)
    if module_name is None:
        return {"category": category, "error": f"Unknown category '{category}'."}

    script_path = os.path.join(
        _get_standalone_dir(), "category_analyses", f"{module_name}.py"
    )

    result = subprocess.run(
        [sys.executable, script_path, "--output-dir", output_dir],
        capture_output=True, text=True,
    )

    metrics_path = os.path.join(output_dir, "category_data", f"{category}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)

    if result.returncode != 0:
        return {"category": category, "error": f"{module_name} failed (exit {result.returncode})",
                "stderr": result.stderr[-2000:] if result.stderr else ""}

    return {"category": category, "status": "completed", "note": "No metrics JSON produced."}


def _run_all_category_analyses(output_dir: str) -> dict:
    manifest_path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if not os.path.exists(manifest_path):
        return {"error": "category_manifest.json not found. Run prepare_agentic first."}

    with open(manifest_path) as f:
        manifest = json.load(f)

    categories = [c["name"] for c in manifest.get("categories", [])]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as pool:
        futures = {cat: pool.submit(_run_single_category, output_dir, cat)
                   for cat in categories}
        results = {cat: fut.result() for cat, fut in futures.items()}

    try:
        analyses_dir = os.path.join(_get_standalone_dir(), "category_analyses")
        if analyses_dir not in sys.path:
            sys.path.insert(0, analyses_dir)
        from analysis_utils import generate_plot_data
        generate_plot_data(output_dir)
    except Exception as e:
        results["_plot_data_error"] = str(e)

    return {"categories_analyzed": len(results), "results": results}


# ---------------------------------------------------------------------------
# Public: run_full_standalone_analysis
# ---------------------------------------------------------------------------

def run_full_standalone_analysis(trace_path: str, platform: str,
                                 output_dir: str = None,
                                 trace_type: str = "pytorch",
                                 enable_pseudo_ops: bool = True,
                                 cleanup: bool = True) -> dict:
    """Run the complete standalone analysis pipeline in one call.

    Returns everything the AI Agent needs to write standalone_analysis.md.
    If cleanup=True (default), intermediate files are deleted automatically.
    """
    import shutil

    if not output_dir:
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="tracelens_")
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "trace_path": trace_path,
        "platform": platform,
        "output_dir": output_dir,
        "steps": {},
    }

    if not os.path.isfile(trace_path):
        result["error"] = f"Trace file not found: {trace_path}"
        return result

    stat = os.stat(trace_path)
    result["file_size_mb"] = round(stat.st_size / (1024 ** 2), 2)

    # Step 1: generate perf report
    step1 = _generate_perf_report(trace_path, output_dir,
                                  trace_type=trace_type,
                                  enable_pseudo_ops=enable_pseudo_ops)
    result["steps"]["generate_perf_report"] = step1
    if "error" in step1:
        result["error"] = "generate_perf_report failed"
        return result

    # Step 2-5: prepare agentic
    step2 = _prepare_agentic(platform, output_dir,
                             trace_path=trace_path,
                             enable_pseudo_ops=enable_pseudo_ops)
    result["steps"]["prepare_agentic"] = step2
    if "error" in step2:
        result["error"] = "prepare_agentic failed"
        return result

    result["gpu_utilization"] = step2.get("gpu_utilization", {})
    result["categories"] = step2.get("categories", [])

    # Step 3: run all category analyses
    step3 = _run_all_category_analyses(output_dir)
    result["steps"]["run_all_category_analyses"] = {
        "categories_analyzed": step3.get("categories_analyzed", 0),
    }

    # Collect all metrics
    category_data_dir = os.path.join(output_dir, "category_data")
    all_metrics = {}
    if os.path.isdir(category_data_dir):
        for fname in sorted(os.listdir(category_data_dir)):
            if fname.endswith("_metrics.json"):
                cat_name = fname.replace("_metrics.json", "")
                with open(os.path.join(category_data_dir, fname)) as f:
                    all_metrics[cat_name] = json.load(f)
            elif fname == "multi_kernel_data.json":
                with open(os.path.join(category_data_dir, fname)) as f:
                    all_metrics["_multi_kernel_data"] = json.load(f)

    result["all_metrics"] = all_metrics
    result["status"] = "completed"

    # Report generation instructions — included in response so the AI
    # always knows how to write the report regardless of prompt usage
    specs, _ = _get_platform_specs()
    platform_spec = specs.get(platform, {})
    max_tflops = platform_spec.get("max_achievable_tflops", {})
    
    result["report_instructions"] = {
        "action": "IMPORTANT: Create a file named 'standalone_analysis.md' with the analysis report. Use the Write tool to save the file.",
        "output_file": "standalone_analysis.md",
        "platform_specs": {
            "name": platform_spec.get("name", platform),
            "hbm_bw_gbps": platform_spec.get("mem_bw_gbps"),
            "memory_gb": platform_spec.get("memory_gb"),
            "peak_tflops_bf16": max_tflops.get("matrix_bf16"),
            "peak_tflops_fp16": max_tflops.get("matrix_fp16"),
            "peak_tflops_fp8": max_tflops.get("matrix_fp8"),
            "peak_tflops_fp32": max_tflops.get("matrix_fp32"),
        },
        "sections": [
            "Executive Summary — GPU utilization, top 3-5 bottlenecks table, potential speedup",
            "System-Level Optimizations — GPU idle time, compute/comm overlap, memcpy issues (from cpu_idle and multi_kernel metrics)",
            "Compute Kernel Optimizations — per-category analysis sorted by gpu_kernel_time_ms descending (from each category's metrics)",
            "Impact Summary — table: Recommendation | Type (system/kernel_tuning/algorithmic) | Estimated Savings (ms) | Confidence",
            "Appendix — Hardware Specs (include ALL platform_specs values: name, memory, bandwidth, peak TFLOPS for each precision)",
        ],
        "rules": [
            "Use gpu_kernel_time_ms for bottleneck ranking, NOT cpu_duration_ms",
            "Flag efficiency > 100% as [ANOMALY]",
            "If cpu_duration_ms >> gpu_kernel_time_ms (>5x), flag as sync bottleneck",
            "Use vendor-agnostic terminology (GPU kernels, collective communication) except when quoting actual kernel names",
            "Priority icons: P1=red (highest impact), P2=yellow, P3=green",
            "MUST include Appendix section with full hardware specifications",
        ],
    }

    if cleanup:
        shutil.rmtree(output_dir, ignore_errors=True)
        result["output_dir"] = None
        result["cleaned_up"] = True

    return result
