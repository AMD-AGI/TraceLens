###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triage check functions and helpers.

Each check function has the signature:

    def check_xxx(run_dir: str, stream_file: str | None) -> Finding | None

and returns a Finding if the failure mode is detected, or None otherwise.
"""

import gzip
import json
import os
import re


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

class Finding:
    __slots__ = ("tag", "category", "failure_mode", "evidence", "remedy")

    def __init__(self, tag, category, failure_mode, evidence, remedy):
        self.tag = tag
        self.category = category
        self.failure_mode = failure_mode
        self.evidence = evidence
        self.remedy = remedy

    def diag_line(self):
        return f"[{self.tag}] {self.evidence}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_trace_path(run_dir):
    """Try to find the trace path from cache/cmd_prefix.txt or the manifest."""
    cmd_prefix = os.path.join(run_dir, "cache", "cmd_prefix.txt")
    if os.path.isfile(cmd_prefix):
        with open(cmd_prefix) as f:
            content = f.read()
        match = re.search(r"--profile_json_path\s+(\S+)", content)
        if match:
            return match.group(1)

    manifest = os.path.join(run_dir, "category_data", "category_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest) as f:
            data = json.load(f)
        tp = data.get("trace_path")
        if tp:
            return tp
    return None


_SDK_PREFIX_RE = re.compile(r"^\[claude-sdk\]\s*")


def stream_lines(stream_file):
    """Yield lines from a stream file (ndjson, streamJSON, or Hyperloom log)."""
    if not stream_file or not os.path.isfile(stream_file):
        return
    with open(stream_file, errors="replace") as f:
        for line in f:
            yield _SDK_PREFIX_RE.sub("", line)


def stream_contains(stream_file, pattern):
    """Check if any line in the stream matches a regex pattern."""
    regex = re.compile(pattern, re.IGNORECASE)
    for line in stream_lines(stream_file):
        if regex.search(line):
            return True
    return False


def stream_find(stream_file, pattern):
    """Return the first matching line (truncated) or None."""
    regex = re.compile(pattern, re.IGNORECASE)
    for line in stream_lines(stream_file):
        if regex.search(line):
            return line.strip()[:200]
    return None


def _load_manifest(run_dir):
    path = os.path.join(run_dir, "category_data", "category_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_trace_json(trace_path):
    """Parse a trace file (plain or gzipped). Returns parsed data or raises."""
    if trace_path.endswith(".gz"):
        with gzip.open(trace_path, "rt", errors="replace") as f:
            return json.load(f)
    with open(trace_path, errors="replace") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Profiling checks
# ---------------------------------------------------------------------------

def check_trace_missing(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if trace_path and not os.path.exists(trace_path):
        return Finding(
            "DIAG:profiling:TRACE_MISSING", "Profiling",
            "Trace files missing",
            f"Trace file not found: {trace_path}",
            "Check profiling setting in workload, or if trace path passed in correctly",
        )


def check_trace_size(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None
    size = os.path.getsize(trace_path)
    if size < 100_000:
        return Finding(
            "DIAG:profiling:TRACE_SIZE", "Profiling",
            "Trace files too small (< 100KB)",
            f"Trace file is {size:,} bytes — may be empty or warmup-only",
            "Profiling window too short or no GPU ops captured",
        )
    if size > 5_000_000_000:
        return Finding(
            "DIAG:profiling:TRACE_SIZE", "Profiling",
            "Trace files too large (> 5GB)",
            f"Trace file is {size / 1e9:.1f} GB — consider splitting",
            "Too many steps are being analyzed; reduce profiling window",
        )


def check_no_gpu_kernels(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None
    if os.path.getsize(trace_path) > 5_000_000_000:
        return None

    try:
        data = _load_trace_json(trace_path)
        events = data if isinstance(data, list) else data.get("traceEvents", [])
        if not any(e.get("cat") == "kernel" for e in events):
            return Finding(
                "DIAG:profiling:NO_GPU_KERNELS", "Profiling",
                "No GPU kernel events in trace",
                "Trace contains no events with cat='kernel'",
                "Ensure ProfilerActivity.CUDA is enabled in profiler config",
            )
    except (json.JSONDecodeError, OSError):
        pass


def check_capture_missing(run_dir, _stream_file):
    manifest = _load_manifest(run_dir)
    if not manifest:
        return None
    capture_folder = manifest.get("capture_folder_path")
    if capture_folder and (not os.path.isdir(capture_folder) or not os.listdir(capture_folder)):
        return Finding(
            "DIAG:profiling:CAPTURE_MISSING", "Profiling",
            "Graph capture traces missing",
            f"Capture folder missing or empty: {capture_folder}",
            "Collect graph capture traces following Inference_analysis.md instructions",
        )


# ---------------------------------------------------------------------------
# Trace quality checks
# ---------------------------------------------------------------------------

def check_high_idle(run_dir, stream_file):
    manifest = _load_manifest(run_dir)
    if manifest:
        idle = manifest.get("gpu_utilization", {}).get("idle_time_percent", 0)
        if idle > 15:
            return Finding(
                "DIAG:trace_quality:HIGH_IDLE", "Trace Quality",
                "High idle time (> 15% GPU idle)",
                f"GPU idle time is {idle:.1f}%",
                "Investigate CPU bottlenecks, graph capture enablement, host-side overhead",
            )
    if stream_contains(stream_file, r"DIAG:trace_quality:HIGH_IDLE"):
        return Finding(
            "DIAG:trace_quality:HIGH_IDLE", "Trace Quality",
            "High idle time (> 15% GPU idle)",
            "DIAG tag found in agent stream",
            "Investigate CPU bottlenecks, graph capture enablement, host-side overhead",
        )


def check_gpu_graph_replay(run_dir, stream_file):
    if stream_contains(stream_file, r"DIAG:trace_quality:GPU_GRAPH_REPLAY"):
        return Finding(
            "DIAG:trace_quality:GPU_GRAPH_REPLAY", "Trace Quality",
            "GPU Graph Replay detected in default mode",
            "DIAG tag found in agent stream",
            "Switch to inference analysis mode (--analysis_mode inference)",
        )


def check_corrupt_json(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None
    if os.path.getsize(trace_path) > 5_000_000_000:
        return None

    try:
        _load_trace_json(trace_path)
    except json.JSONDecodeError as e:
        return Finding(
            "DIAG:trace_quality:CORRUPT_JSON", "Trace Quality",
            "Trace file is corrupted/invalid JSON",
            f"JSONDecodeError: {str(e)[:150]}",
            "Re-collect the trace; verify .json.gz files are not truncated",
        )
    except OSError as e:
        return Finding(
            "DIAG:trace_quality:CORRUPT_JSON", "Trace Quality",
            "Trace file is corrupted/invalid JSON",
            f"OSError reading trace: {str(e)[:150]}",
            "Re-collect the trace; verify .json.gz files are not truncated",
        )


# ---------------------------------------------------------------------------
# Pipeline checks
# ---------------------------------------------------------------------------

def check_step1_fail(run_dir, stream_file):
    csv_dir = os.path.join(run_dir, "perf_report_csvs")
    if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
        return Finding(
            "DIAG:pipeline:STEP1_FAIL", "Pipeline",
            "Performance report generation fails (Step 1)",
            f"perf_report_csvs/ missing or empty at {csv_dir}",
            "Check trace file path; verify platform JSON exists",
        )
    if stream_contains(stream_file, r"DIAG:pipeline:STEP1_FAIL"):
        return Finding(
            "DIAG:pipeline:STEP1_FAIL", "Pipeline",
            "Performance report generation fails (Step 1)",
            "DIAG tag found in agent stream",
            "Check trace file path; verify platform JSON exists",
        )


def check_step2_5_fail(run_dir, stream_file):
    manifest_path = os.path.join(run_dir, "category_data", "category_manifest.json")
    if os.path.isdir(os.path.join(run_dir, "perf_report_csvs")) and not os.path.isfile(manifest_path):
        return Finding(
            "DIAG:pipeline:STEP2_5_FAIL", "Pipeline",
            "Orchestrator preparation fails (Steps 2-5)",
            "category_manifest.json missing despite perf_report_csvs/ existing",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )
    if stream_contains(stream_file, r"DIAG:pipeline:STEP2_5_FAIL"):
        return Finding(
            "DIAG:pipeline:STEP2_5_FAIL", "Pipeline",
            "Orchestrator preparation fails (Steps 2-5)",
            "DIAG tag found in agent stream",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )


def check_output_incomplete(run_dir, _stream_file):
    expected = ["metadata", "category_data", "system_findings", "category_findings"]
    missing = [d for d in expected if not os.path.isdir(os.path.join(run_dir, d))]
    if missing:
        return Finding(
            "DIAG:pipeline:OUTPUT_INCOMPLETE", "Pipeline",
            "Output directory structure incomplete",
            f"Missing directories: {', '.join(missing)}",
            "Re-run analysis pipeline from Step 1; check disk space and write permissions",
        )


def check_prefix_fail(_run_dir, stream_file):
    if stream_contains(stream_file, r"DIAG:pipeline:PREFIX_FAIL"):
        return Finding(
            "DIAG:pipeline:PREFIX_FAIL", "Pipeline",
            "Command prefix validation fails",
            "DIAG tag found in agent stream",
            "Verify TraceLens is installed; check SSH/container/venv connectivity",
        )


# ---------------------------------------------------------------------------
# Report checks
# ---------------------------------------------------------------------------

def check_report_too_small(run_dir, _stream_file):
    report = os.path.join(run_dir, "analysis.md")
    if os.path.isfile(report) and os.path.getsize(report) < 100:
        return Finding(
            "DIAG:report:REPORT_TOO_SMALL", "Report",
            "analysis.md too small (< 100 bytes)",
            f"analysis.md is {os.path.getsize(report)} bytes",
            "Report generation likely crashed mid-write",
        )
    if not os.path.isfile(report) and os.path.isdir(os.path.join(run_dir, "category_findings")):
        return Finding(
            "DIAG:report:REPORT_TOO_SMALL", "Report",
            "analysis.md missing",
            "analysis.md does not exist but category_findings/ is present",
            "Report generation likely crashed or was interrupted",
        )


# ---------------------------------------------------------------------------
# Token budget checks
# ---------------------------------------------------------------------------

def check_resource_exhausted(_run_dir, stream_file):
    for line in stream_lines(stream_file):
        stripped = line.strip()
        try:
            record = json.loads(stripped)
            rtype = record.get("type", "")
            if rtype == "error" and "RESOURCE_EXHAUSTED" in json.dumps(record):
                return Finding(
                    "DIAG:token:RESOURCE_EXHAUSTED", "Token Budget",
                    "Context length exceeded (RESOURCE_EXHAUSTED)",
                    "RESOURCE_EXHAUSTED error event in agent stream",
                    "Reduce trace size before analysis",
                )
            if rtype == "system" and "RESOURCE_EXHAUSTED" in str(record.get("error", "")):
                return Finding(
                    "DIAG:token:RESOURCE_EXHAUSTED", "Token Budget",
                    "Context length exceeded (RESOURCE_EXHAUSTED)",
                    "RESOURCE_EXHAUSTED system error in agent stream",
                    "Reduce trace size before analysis",
                )
        except (json.JSONDecodeError, ValueError):
            if re.search(r"resource_exhausted", stripped, re.IGNORECASE):
                return Finding(
                    "DIAG:token:RESOURCE_EXHAUSTED", "Token Budget",
                    "Context length exceeded (RESOURCE_EXHAUSTED)",
                    f"resource_exhausted in stream: {stripped[:150]}",
                    "Reduce trace size before analysis",
                )


def check_subagent_budget(run_dir, _stream_file):
    manifest = _load_manifest(run_dir)
    if not manifest:
        return None
    findings_dir = os.path.join(run_dir, "category_findings")
    system_dir = os.path.join(run_dir, "system_findings")

    missing = []
    for cat in manifest.get("categories", []):
        name = cat.get("name", "")
        if not name:
            continue
        tier = cat.get("tier", "compute_kernel")
        target_dir = system_dir if tier == "system" else findings_dir
        if not os.path.isfile(os.path.join(target_dir, f"{name}_findings.md")):
            missing.append(name)

    if missing:
        return Finding(
            "DIAG:token:SUBAGENT_BUDGET", "Token Budget",
            "Subagent exceeds token budget",
            f"Missing findings for: {', '.join(missing)}",
            "Check token budget; subagent(s) may have terminated without writing output",
        )


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------

def check_ssh_fail(_run_dir, stream_file):
    match = stream_find(stream_file, r"(ssh:.*(?:timeout|timed out)|Connection refused|Permission denied \(publickey)")
    if match:
        return Finding(
            "DIAG:infra:SSH_FAIL", "Infrastructure",
            "SSH connection to node fails",
            match,
            "Verify node hostname, SSH keys, and network connectivity",
        )


def check_docker_missing(_run_dir, stream_file):
    match = stream_find(stream_file, r"(Error: No such container|Error response from daemon:.*not running)")
    if match:
        return Finding(
            "DIAG:infra:DOCKER_MISSING", "Infrastructure",
            "Docker container not found",
            match,
            "Start the container or verify container name; check docker ps",
        )


def check_tl_not_installed(_run_dir, stream_file):
    match = stream_find(stream_file, r"(ModuleNotFoundError: No module named 'TraceLens'|No module named 'TraceLens')")
    if match:
        return Finding(
            "DIAG:infra:TL_NOT_INSTALLED", "Infrastructure",
            "TraceLens not installed in remote env",
            match,
            "Run pip install git+https://github.com/AMD-AGI/TraceLens.git",
        )


def check_disk_full(run_dir, stream_file):
    zero_byte = []
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.getsize(fpath) == 0 and not fname.startswith("."):
                zero_byte.append(os.path.relpath(fpath, run_dir))
    if len(zero_byte) > 3:
        return Finding(
            "DIAG:infra:DISK_FULL", "Infrastructure",
            "Disk space exhausted",
            f"{len(zero_byte)} zero-byte files found (e.g. {zero_byte[0]})",
            "Free disk space on the target node",
        )
    if stream_contains(stream_file, r"No space left on device"):
        return Finding(
            "DIAG:infra:DISK_FULL", "Infrastructure",
            "Disk space exhausted",
            "'No space left on device' found in agent stream",
            "Free disk space on the target node",
        )


def check_nfs_stale(_run_dir, stream_file):
    match = stream_find(stream_file, r"(Stale file handle|stale NFS|NFS.*stale)")
    if match:
        return Finding(
            "DIAG:infra:NFS_STALE", "Infrastructure",
            "NFS latency / stale file handles",
            match,
            "Add small delays between write and read operations; verify NFS mount",
        )


def check_missing_dep(_run_dir, stream_file):
    match = stream_find(stream_file, r"(ImportError: No module named|ModuleNotFoundError: No module named)(?!.*TraceLens)")
    if match:
        return Finding(
            "DIAG:infra:MISSING_DEP", "Infrastructure",
            "Python dependency missing",
            match,
            "Install TraceLens with all dependencies",
        )


# ---------------------------------------------------------------------------
# Check registry (order follows the CSV)
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_trace_missing,
    check_trace_size,
    check_no_gpu_kernels,
    check_capture_missing,
    check_high_idle,
    check_gpu_graph_replay,
    check_corrupt_json,
    check_step1_fail,
    check_step2_5_fail,
    check_output_incomplete,
    check_prefix_fail,
    check_report_too_small,
    check_resource_exhausted,
    check_subagent_budget,
    check_ssh_fail,
    check_docker_missing,
    check_tl_not_installed,
    check_disk_full,
    check_nfs_stale,
    check_missing_dep,
]
