###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Profiling harness for TraceLens nightly performance dashboard.

Wraps TraceLens report generation in cProfile, extracts per-stage cumtime
via pstats, records peak RSS, outputs structured timing JSON, and optionally
emits OTLP metrics to Grafana Cloud.

Usage:
    # Single trace, local only
    python scripts/tracelens_perf_harness.py \
        --trace-file /path/to/trace.json \
        --output-dir ./perf_results

    # Manifest-driven, with OTLP emission
    python scripts/tracelens_perf_harness.py \
        --manifest config/trace_manifest.yaml \
        --trace-dir /tmp/traces \
        --output-dir ./perf_results \
        --emit-otlp

    # Manifest with filter
    python scripts/tracelens_perf_harness.py \
        --manifest config/trace_manifest.yaml \
        --trace-dir /tmp/traces \
        --output-dir ./perf_results \
        --filter "trace_001,trace_003"
"""

import argparse
import cProfile
import json
import os
import platform
import pstats
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add project root to path so TraceLens can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from TraceLens.TreePerf import TreePerfAnalyzer

# Stage function names to extract from pstats (matched on function name).
# These correspond to the 12 stages tracked in the dashboard.
STAGES = {
    "total_report_generation",
    "from_file",
    "load_data",
    "build_tree",
    "build_host_call_stack_tree",
    "label_non_gpu_paths",
    "add_gpu_ops_to_tree",
    "collect_unified_perf_events",
    "build_df_unified_perf_table",
    "get_df_kernel_launchers",
    "get_df_kernels",
    "get_df_gpu_timeline",
}


def total_report_generation(trace_path):
    """Top-level wrapper that runs the full TraceLens pipeline for a trace.

    Named so that pstats records it as the 'total_report_generation' stage.
    """
    analyzer = TreePerfAnalyzer.from_file(trace_path)
    analyzer.build_df_unified_perf_table()
    analyzer.get_df_kernel_launchers()
    analyzer.get_df_kernels()
    analyzer.get_df_gpu_timeline()
    return analyzer


def run_tracelens_with_cprofile(trace_path, artifact_path):
    """Run TraceLens report generation under cProfile and save the .prof artifact."""
    pr = cProfile.Profile()
    pr.enable()
    total_report_generation(trace_path)
    pr.disable()
    pr.dump_stats(artifact_path)
    return pr


def extract_stage_timings(prof):
    """Parse the in-memory cProfile.Profile object and return cumtime for each tracked stage."""
    stats = pstats.Stats(prof)
    timings = {}
    for (_, _, func), (_, _, _, cumtime, _) in stats.stats.items():
        if func in STAGES:
            timings[func] = round(cumtime, 4)
    return timings


def profile_trace(trace_path, trace_id, output_dir):
    """Run TraceLens under cProfile, extract stage timings, and record peak memory."""
    artifact_path = os.path.join(output_dir, f"{trace_id}_profile.prof")
    pr = run_tracelens_with_cprofile(trace_path, artifact_path)
    timings = extract_stage_timings(pr)
    max_rss_bytes = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )  # KB -> bytes
    return {
        "stages": timings,
        "max_rss_bytes": max_rss_bytes,
        "cprofile_artifact": artifact_path,
    }


def load_manifest(manifest_path):
    """Load trace manifest YAML and return list of trace entries."""
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    return manifest.get("traces", [])


def get_tracelens_version():
    """Attempt to read TraceLens version from setup.py or package metadata."""
    try:
        from importlib.metadata import version

        return version("TraceLens")
    except Exception:
        return "unknown"


def get_commit_sha():
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def build_metadata():
    """Build metadata dict for the timing JSON output."""
    return {
        "commit_sha": get_commit_sha(),
        "run_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "tracelens_version": get_tracelens_version(),
        "environment": os.environ.get("GITHUB_ACTIONS", "local"),
    }


def emit_otlp_metrics(results, metadata):
    """Push stage timings and memory metrics to Grafana Cloud via OTLP/HTTP."""
    from opentelemetry import metrics
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.resources import Resource

    otlp_endpoint = os.environ.get("GRAFANA_CLOUD_OTLP_ENDPOINT")
    push_token = os.environ.get("GRAFANA_CLOUD_OTLP_TOKEN")

    if not otlp_endpoint:
        print("Warning: GRAFANA_CLOUD_OTLP_ENDPOINT not set, skipping OTLP emission")
        return

    otel_resource = Resource.create(
        {
            "service.name": "tracelens.perf_harness",
            "service.version": metadata["tracelens_version"],
            "vcs.commit.sha": metadata["commit_sha"],
        }
    )

    headers = {}
    if push_token:
        headers["Authorization"] = f"Bearer {push_token}"

    # Append /v1/metrics explicitly so the SDK does not double-append the suffix
    if not otlp_endpoint.endswith("/v1/metrics"):
        otlp_endpoint = otlp_endpoint.rstrip("/") + "/v1/metrics"
    exporter = OTLPMetricExporter(endpoint=otlp_endpoint, headers=headers)

    # InMemoryMetricReader has no background thread — it holds metrics in memory
    # and only exports when we explicitly call collect() + exporter.export().
    reader = InMemoryMetricReader()
    provider = MeterProvider(resource=otel_resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    meter = metrics.get_meter("tracelens.perf_harness")
    stage_gauge = meter.create_gauge(
        name="tracelens.stage.duration_seconds",
        description="cumtime for a TraceLens processing stage",
        unit="s",
    )
    total_gauge = meter.create_gauge(
        name="tracelens.total.duration_seconds",
        description="Total end-to-end processing time",
        unit="s",
    )
    rss_gauge = meter.create_gauge(
        name="tracelens.process.max_rss_bytes",
        description="Peak resident memory after processing",
        unit="By",
    )

    for trace_result in results:
        trace_id = trace_result["trace_id"]
        workload_family = trace_result.get("workload_family", "unknown")

        for stage, duration in trace_result["stages"].items():
            stage_gauge.set(
                duration,
                attributes={
                    "trace_id": trace_id,
                    "stage": stage,
                    "workload_family": workload_family,
                },
            )

        total_duration = trace_result["stages"].get("total_report_generation")
        if total_duration is not None:
            total_gauge.set(
                total_duration,
                attributes={
                    "trace_id": trace_id,
                    "workload_family": workload_family,
                },
            )

        rss_gauge.set(
            trace_result["max_rss_bytes"],
            attributes={
                "trace_id": trace_id,
                "workload_family": workload_family,
            },
        )

    # All gauges are set — collect from memory and export exactly once.
    metrics_data = reader.get_metrics_data()
    exporter.export(metrics_data)
    provider.shutdown()
    print("OTLP metrics emitted successfully")
    print("OTLP metrics emitted successfully")


def run_manifest(manifest_path, trace_dir, output_dir, filter_ids=None):
    """Profile all enabled traces from a manifest and return list of result dicts."""
    traces = load_manifest(manifest_path)
    results = []

    for trace_entry in traces:
        if not trace_entry.get("enabled", True):
            continue

        tid = trace_entry["trace_id"]
        if filter_ids and tid not in filter_ids:
            continue

        trace_filename = Path(trace_entry["source_location"]).name
        trace_path = os.path.join(trace_dir, trace_filename)

        if not os.path.exists(trace_path):
            print(f"Warning: trace file not found: {trace_path}, skipping {tid}")
            continue

        print(f"Profiling trace: {tid}")
        result = profile_trace(trace_path, tid, output_dir)
        result["trace_id"] = tid
        result["workload_family"] = trace_entry.get("workload_family", "unknown")
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TraceLens profiling harness for nightly performance dashboard"
    )
    parser.add_argument(
        "--manifest",
        help="Path to trace manifest YAML",
    )
    parser.add_argument(
        "--trace-dir",
        help="Directory containing trace files",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write timing JSON and cProfile artifacts",
    )
    parser.add_argument(
        "--filter",
        help="Comma-separated list of trace_ids to include (used with --manifest)",
    )
    parser.add_argument(
        "--emit-otlp",
        action="store_true",
        help="Emit metrics to Grafana Cloud via OTLP/HTTP",
    )
    parser.add_argument(
        "--push-only",
        metavar="TIMING_JSON",
        help="Skip profiling and push an existing timing.json to Grafana Cloud",
    )

    args = parser.parse_args()

    if args.push_only:
        with open(args.push_only) as f:
            data = json.load(f)
        emit_otlp_metrics(data["traces"], data["metadata"])
        return

    if not args.manifest or not args.trace_dir or not args.output_dir:
        parser.error("--manifest, --trace-dir, and --output-dir are required")

    os.makedirs(args.output_dir, exist_ok=True)

    filter_ids = None
    if args.filter:
        filter_ids = set(args.filter.split(","))

    results = run_manifest(args.manifest, args.trace_dir, args.output_dir, filter_ids)

    if not results:
        print("No traces were profiled")
        sys.exit(1)

    metadata = build_metadata()
    output = {"metadata": metadata, "traces": results}
    timing_path = os.path.join(args.output_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Timing JSON written to {timing_path}")

    if args.emit_otlp:
        emit_otlp_metrics(results, metadata)


if __name__ == "__main__":
    main()
