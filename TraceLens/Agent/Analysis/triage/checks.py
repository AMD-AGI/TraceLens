###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triage check functions and helpers.

Each check function has the signature:

    def check_xxx(run_dir: str, stream_file: str | None) -> FindingDraft | None

and returns a FindingDraft if the failure mode is detected, or None otherwise.
The DIAG tag, category, and sublabel are owned by the check's CheckSpec entry
in ``ALL_CHECKS``; the runner builds the final ``Finding`` from spec + draft.
"""

import glob
import gzip
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Finding / FindingDraft / CheckSpec
# ---------------------------------------------------------------------------

@dataclass
class FindingDraft:
    """Per-check output: just the descriptive fields. The runner attaches the
    DIAG tag, category, and sublabel from the matching CheckSpec."""
    failure_mode: str
    evidence: str
    remedy: str


@dataclass
class Finding:
    tag: str
    sublabel: str
    category: str
    failure_mode: str
    evidence: str
    remedy: str
    implied_by: List[str] = field(default_factory=list)

    def diag_line(self):
        suffix = f" (implied by {', '.join(self.implied_by)})" if self.implied_by else ""
        return f"[{self.tag}] {self.evidence}{suffix}"


@dataclass(frozen=True)
class CheckSpec:
    """Metadata + function pointer for a single triage check.

    Attributes:
        sublabel: Short ordinal label such as "1a", "1b". May be "" until a
            label has been assigned, in which case the DIAG tag falls back to
            ``DIAG:{category}:{name}``.
        name: NAME slot of the DIAG tag, e.g. "TRACE_MISSING".
        category: Short category prefix, e.g. "profiling", "trace_quality",
            "pipeline", "report", "token", "infra".
        fn: The check function. Signature ``(run_dir, stream_file) -> FindingDraft | None``.
        implies_failures: Sublabels of other checks that will definitely fail
            if this check fails. Used by the runner to annotate downstream
            findings with ``implied_by``. Safe to leave empty.
        detailed_only: If True, this check runs only when the runner is
            invoked with ``--detailed``. Use for expensive or noisy checks
            that aren't useful in the default fast pass.
    """
    sublabel: str
    name: str
    category: str
    fn: Callable[[str, Optional[str]], Optional[FindingDraft]]
    implies_failures: List[str] = field(default_factory=list)
    detailed_only: bool = False

    def build_tag(self) -> str:
        if self.sublabel:
            return f"DIAG:{self.category}:{self.sublabel}_{self.name}"
        return f"DIAG:{self.category}:{self.name}"


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
    torch_trace_dir = os.path.join(run_dir, "torch_trace")
    if os.path.isdir(torch_trace_dir):
        gz_files = glob.glob(os.path.join(torch_trace_dir, "*.trace.json.gz"))
        if not gz_files:
            return FindingDraft(
                "Trace files missing",
                f"torch_trace/ exists but contains no *.trace.json.gz files: {torch_trace_dir}",
                "Check profiling setting in workload, or if trace path passed in correctly",
            )

    trace_path = resolve_trace_path(run_dir)
    if trace_path and not os.path.exists(trace_path):
        return FindingDraft(
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
        return FindingDraft(
            "Trace files too small (< 100KB)",
            f"Trace file is {size:,} bytes — may be empty or warmup-only",
            "Profiling window too short or no GPU ops captured",
        )
    if size > 5_000_000_000:
        return FindingDraft(
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
            return FindingDraft(
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
        return FindingDraft(
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
            return FindingDraft(
                "High idle time (> 15% GPU idle)",
                f"GPU idle time is {idle:.1f}%",
                "Investigate CPU bottlenecks, graph capture enablement, host-side overhead",
            )
    if stream_contains(stream_file, r"DIAG:trace_quality:HIGH_IDLE"):
        return FindingDraft(
            "High idle time (> 15% GPU idle)",
            "DIAG tag found in agent stream",
            "Investigate CPU bottlenecks, graph capture enablement, host-side overhead",
        )


def check_gpu_graph_replay(_run_dir, stream_file):
    if stream_contains(stream_file, r"DIAG:trace_quality:GPU_GRAPH_REPLAY"):
        return FindingDraft(
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
        return FindingDraft(
            "Trace file is corrupted/invalid JSON",
            f"JSONDecodeError: {str(e)[:150]}",
            "Re-collect the trace; verify .json.gz files are not truncated",
        )
    except OSError as e:
        return FindingDraft(
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
        return FindingDraft(
            "Performance report generation fails (Step 1)",
            f"perf_report_csvs/ missing or empty at {csv_dir}",
            "Check trace file path; verify platform JSON exists",
        )
    if stream_contains(stream_file, r"DIAG:pipeline:STEP1_FAIL"):
        return FindingDraft(
            "Performance report generation fails (Step 1)",
            "DIAG tag found in agent stream",
            "Check trace file path; verify platform JSON exists",
        )


def check_step2_5_fail(run_dir, stream_file):
    manifest_path = os.path.join(run_dir, "category_data", "category_manifest.json")
    if os.path.isdir(os.path.join(run_dir, "perf_report_csvs")) and not os.path.isfile(manifest_path):
        return FindingDraft(
            "Orchestrator preparation fails (Steps 2-5)",
            "category_manifest.json missing despite perf_report_csvs/ existing",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )
    if stream_contains(stream_file, r"DIAG:pipeline:STEP2_5_FAIL"):
        return FindingDraft(
            "Orchestrator preparation fails (Steps 2-5)",
            "DIAG tag found in agent stream",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )


def check_output_incomplete(run_dir, _stream_file):
    expected = ["metadata", "category_data", "system_findings", "category_findings"]
    missing = [d for d in expected if not os.path.isdir(os.path.join(run_dir, d))]
    if missing:
        return FindingDraft(
            "Output directory structure incomplete",
            f"Missing directories: {', '.join(missing)}",
            "Re-run analysis pipeline from Step 1; check disk space and write permissions",
        )


def check_prefix_fail(_run_dir, stream_file):
    if stream_contains(stream_file, r"DIAG:pipeline:PREFIX_FAIL"):
        return FindingDraft(
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
        return FindingDraft(
            "analysis.md too small (< 100 bytes)",
            f"analysis.md is {os.path.getsize(report)} bytes",
            "Report generation likely crashed mid-write",
        )
    if not os.path.isfile(report) and os.path.isdir(os.path.join(run_dir, "category_findings")):
        return FindingDraft(
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
                return FindingDraft(
                    "Context length exceeded (RESOURCE_EXHAUSTED)",
                    "RESOURCE_EXHAUSTED error event in agent stream",
                    "Reduce trace size before analysis",
                )
            if rtype == "system" and "RESOURCE_EXHAUSTED" in str(record.get("error", "")):
                return FindingDraft(
                    "Context length exceeded (RESOURCE_EXHAUSTED)",
                    "RESOURCE_EXHAUSTED system error in agent stream",
                    "Reduce trace size before analysis",
                )
        except (json.JSONDecodeError, ValueError):
            if re.search(r"resource_exhausted", stripped, re.IGNORECASE):
                return FindingDraft(
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
        return FindingDraft(
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
        return FindingDraft(
            "SSH connection to node fails",
            match,
            "Verify node hostname, SSH keys, and network connectivity",
        )


def check_docker_missing(_run_dir, stream_file):
    match = stream_find(stream_file, r"(Error: No such container|Error response from daemon:.*not running)")
    if match:
        return FindingDraft(
            "Docker container not found",
            match,
            "Start the container or verify container name; check docker ps",
        )


def check_tl_not_installed(_run_dir, stream_file):
    match = stream_find(stream_file, r"(ModuleNotFoundError: No module named 'TraceLens'|No module named 'TraceLens')")
    if match:
        return FindingDraft(
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
        return FindingDraft(
            "Disk space exhausted",
            f"{len(zero_byte)} zero-byte files found (e.g. {zero_byte[0]})",
            "Free disk space on the target node",
        )
    if stream_contains(stream_file, r"No space left on device"):
        return FindingDraft(
            "Disk space exhausted",
            "'No space left on device' found in agent stream",
            "Free disk space on the target node",
        )


def check_nfs_stale(_run_dir, stream_file):
    match = stream_find(stream_file, r"(Stale file handle|stale NFS|NFS.*stale)")
    if match:
        return FindingDraft(
            "NFS latency / stale file handles",
            match,
            "Add small delays between write and read operations; verify NFS mount",
        )


def check_missing_dep(_run_dir, stream_file):
    match = stream_find(stream_file, r"(ImportError: No module named|ModuleNotFoundError: No module named)(?!.*TraceLens)")
    if match:
        return FindingDraft(
            "Python dependency missing",
            match,
            "Install TraceLens with all dependencies",
        )


# ---------------------------------------------------------------------------
# Check registry (order follows the CSV)
#
# Sublabels are filled in as the finalized list is reviewed. Empty sublabel
# means "not yet labeled"; the runner falls back to the legacy
# DIAG:<category>:<NAME> tag in that case.
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    CheckSpec("1a", "TRACE_MISSING",      "profiling",     check_trace_missing,
              implies_failures=[]),
    CheckSpec("1b", "TRACE_SIZE",         "profiling",     check_trace_size),
    CheckSpec("1c", "NO_GPU_KERNELS",     "profiling",     check_no_gpu_kernels),
    CheckSpec("1d", "CAPTURE_MISSING",    "profiling",     check_capture_missing),
    CheckSpec("",   "HIGH_IDLE",          "trace_quality", check_high_idle),
    CheckSpec("",   "GPU_GRAPH_REPLAY",   "trace_quality", check_gpu_graph_replay),
    CheckSpec("",   "CORRUPT_JSON",       "trace_quality", check_corrupt_json),
    CheckSpec("",   "STEP1_FAIL",         "pipeline",      check_step1_fail),
    CheckSpec("",   "STEP2_5_FAIL",       "pipeline",      check_step2_5_fail),
    CheckSpec("",   "OUTPUT_INCOMPLETE",  "pipeline",      check_output_incomplete),
    CheckSpec("",   "PREFIX_FAIL",        "pipeline",      check_prefix_fail),
    CheckSpec("",   "REPORT_TOO_SMALL",   "report",        check_report_too_small),
    CheckSpec("",   "RESOURCE_EXHAUSTED", "token",         check_resource_exhausted),
    CheckSpec("",   "SUBAGENT_BUDGET",    "token",         check_subagent_budget),
    CheckSpec("",   "SSH_FAIL",           "infra",         check_ssh_fail),
    CheckSpec("",   "DOCKER_MISSING",     "infra",         check_docker_missing),
    CheckSpec("",   "TL_NOT_INSTALLED",   "infra",         check_tl_not_installed),
    CheckSpec("",   "DISK_FULL",          "infra",         check_disk_full),
    CheckSpec("",   "NFS_STALE",          "infra",         check_nfs_stale),
    CheckSpec("",   "MISSING_DEP",        "infra",         check_missing_dep),
]
