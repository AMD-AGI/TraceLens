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

import csv
import functools
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from TraceLens.util import DataLoader

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
        suffix = (
            f" (implied by {', '.join(self.implied_by)})" if self.implied_by else ""
        )
        return f"[{self.tag}] {self.evidence}{suffix}"


@dataclass(frozen=True)
class CheckSpec:
    """Metadata + function pointer for a single triage check.

    Attributes:
        sublabel: Ordinal label such as "1a", "2f", "4b". Every check must
            have a sublabel; the DIAG tag is ``DIAG:{category}:{sublabel}_{name}``.
        name: NAME slot of the DIAG tag, e.g. "TRACE_MISSING".
        category: Short category prefix, e.g. "profiling", "trace_quality",
            "perf_model", "tracelens_agent_workflow", "infra".
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

_PATH_REMAPS_ENV = "TRACELENS_PATH_REMAPS"
_PATH_REMAPS_CACHE = None
_SDK_PREFIX_RE = re.compile(r"^\[claude-sdk\]\s*")
_TBS_TFLOPS_EXEMPT_SUBSTRINGS = ("collective", "custom")

# Trace files larger than this (bytes) are skipped by detection checks: parsing
# multi-GB traces in-process is too costly for a fast triage pass.
_MAX_TRACE_BYTES = 5_000_000_000
# Traces below this (bytes) are treated as empty / warmup-only.
_MIN_TRACE_BYTES = 100_000
# "Significant op" band for unified_perf_summary rows.
_SIGNIFICANT_CUM_PCT_MAX = 90
_SIGNIFICANT_PCT_MIN = 4
# Kernel-time coefficient-of-variation (std/mean) above which a run is unstable.
_RUNTIME_INSTABILITY_CV = 0.25
# kernel/cpu_op ratio below which a split trace is flagged as missing GPU kernels.
_SPLIT_LOW_KERNEL_RATIO = 0.1
# GPU idle percentage above which idle time is flagged.
_HIGH_IDLE_PCT = 15
# Minimum cpu_op events carrying input shapes before a capture is deemed OK.
_MIN_CPU_OP_SHAPES = 10
# Zero-byte file count above which disk exhaustion is suspected.
_MAX_ZERO_BYTE_FILES = 3


def _path_remaps():
    """Return the list of ``(old_prefix, new_prefix)`` remaps from the environment.

    The ``TRACELENS_PATH_REMAPS`` env var holds comma-separated ``old=new`` pairs,
    applied to absolute paths read from JSON / command files when the original
    path doesn't exist on the local filesystem. Example::

        TRACELENS_PATH_REMAPS=/old_root/=/new_root/,/legacy/=/current/
    """
    global _PATH_REMAPS_CACHE
    if _PATH_REMAPS_CACHE is not None:
        return _PATH_REMAPS_CACHE

    raw = os.environ.get(_PATH_REMAPS_ENV, "")
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        old, _, new = chunk.partition("=")
        old = old.strip()
        new = new.strip()
        if old:
            pairs.append((old, new))
    _PATH_REMAPS_CACHE = pairs
    return pairs


def _apply_path_remaps(path):
    """Yield candidate remapped paths for ``path`` using ``_path_remaps()``."""
    if not path:
        return
    for old, new in _path_remaps():
        if path.startswith(old):
            yield new + path[len(old) :]


def _resolve_path(path, predicate):
    """Return ``path`` (or a remapped variant) if ``predicate`` holds, else None.

    Candidate alternates come from ``TRACELENS_PATH_REMAPS`` (see ``_path_remaps``).
    """
    if not path:
        return None
    if predicate(path):
        return path
    for candidate in _apply_path_remaps(path):
        if predicate(candidate):
            return candidate
    return None


def resolve_trace_path(run_dir):
    """Try to find the trace path from cache/cmd_prefix.txt or the manifest.

    Paths stored in JSON / command files may have been written from a host that
    uses a different absolute filesystem root than the reader. Set
    ``TRACELENS_PATH_REMAPS`` (see ``_path_remaps``) to translate those paths.
    """
    cmd_prefix = os.path.join(run_dir, "cache", "cmd_prefix.txt")
    if os.path.isfile(cmd_prefix):
        with open(cmd_prefix) as f:
            content = f.read()
        match = re.search(r"--profile_json_path\s+(\S+)", content)
        if match:
            resolved = _resolve_path(match.group(1), os.path.isfile)
            if resolved:
                return resolved

    data = _load_manifest(run_dir)
    if data:
        tp = data.get("trace_path")
        if tp:
            return _resolve_path(tp, os.path.isfile) or tp
    return None


def _trace_input_dir(run_dir):
    """Resolve the ``trace_input`` directory from the sibling ``trace_input_manifest.json``.

    The manifest sits one level above ``run_dir``; returns the resolved input
    directory or None if the manifest is missing, unreadable, or its path can't
    be resolved.
    """
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(run_dir)), "trace_input_manifest.json"
    )
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _resolve_path(data.get("trace_input"), os.path.isdir)


def resolve_main_trace_path(run_dir):
    """Resolve the main rank-0 trace file for a run."""
    trace_input = _trace_input_dir(run_dir)
    if trace_input:
        for pattern in ("*TP-0*.json*", "*rank0*.json*"):
            matches = sorted(glob.glob(os.path.join(trace_input, pattern)))
            if matches:
                return matches[0]

    # Fallback: search within this run only.
    for search_root in (os.path.join(run_dir, "trace_split"), run_dir):
        if not os.path.isdir(search_root):
            continue
        for pattern in ("*TP-0*.json*", "*rank0*.json*"):
            matches = sorted(
                glob.glob(os.path.join(search_root, "**", pattern), recursive=True)
            )
            if matches:
                return matches[0]
    return None


def resolve_capture_folder(run_dir):
    """Resolve the ``capture_traces`` folder for a run."""
    trace_input = _trace_input_dir(run_dir)
    if not trace_input:
        return None
    return _resolve_path(os.path.join(trace_input, "capture_traces"), os.path.isdir)


def stream_lines(stream_file):
    """Yield lines from a stream file (ndjson, streamJSON, or plain log)."""
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


@functools.lru_cache(maxsize=1)
def _load_trace_json(trace_path):
    """Parse a trace file (plain or gzipped) via the library loader. Returns
    parsed data or raises.

    Delegates to ``DataLoader.load_data`` (orjson, strict UTF-8) and caches the
    single most-recent parse so checks resolving the same rank-0 file don't
    re-parse a multi-GB trace. ``orjson.JSONDecodeError`` subclasses
    ``json.JSONDecodeError``, so callers guarding ``(OSError, json.JSONDecodeError)``
    still catch parse failures; ``lru_cache`` does not memoize exceptions, so a
    corrupt file re-raises on every call.
    """
    return DataLoader.load_data(trace_path)


def _events_of(data):
    """Extract the trace-event list from parsed trace data.

    Traces are either a bare event list or a dict with a ``traceEvents`` key; a
    dict without that key yields ``[]`` (empty, not an error).
    """
    return data if isinstance(data, list) else data.get("traceEvents", [])


def _load_events(trace_path):
    """Resolve, size-guard, load, and extract events for a detection check.

    Returns the event list, or ``None`` when the file is absent, too large, or
    fails to parse. Error-*reporting* checks that must distinguish missing from
    corrupt call ``_load_trace_json`` directly instead.
    """
    if not trace_path or not os.path.exists(trace_path):
        return None
    if os.path.getsize(trace_path) > _MAX_TRACE_BYTES:
        return None
    try:
        return _events_of(_load_trace_json(trace_path))
    except (OSError, ValueError):
        return None


def resolve_split_trace_dir(run_dir):
    """Return ``<run_dir>/trace_split`` if it exists, else None."""
    path = os.path.join(run_dir, "trace_split")
    return path if os.path.isdir(path) else None


def _load_unified_perf_summary(run_dir):
    """Load ``perf_report_csvs/unified_perf_summary.csv`` as a list of dicts.

    Returns ``None`` if the file is missing or unreadable.
    """
    path = os.path.join(run_dir, "perf_report_csvs", "unified_perf_summary.csv")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None


def _load_gpu_timeline(run_dir):
    """Parse ``perf_report_csvs/gpu_timeline.csv`` into ``{type: percent}``.

    Returns ``None`` if the file is missing or unreadable.
    """
    path = os.path.join(run_dir, "perf_report_csvs", "gpu_timeline.csv")
    if not os.path.isfile(path):
        return None
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                t = (row.get("type") or "").strip()
                pct = _parse_float(row.get("percent"))
                if not t or pct is None:
                    continue
                out[t] = pct
    except (OSError, csv.Error):
        return None
    return out


def _significant_rows(rows):
    """Filter unified_perf_summary rows to "significant ops".

    Significant = ``Cumulative Percentage (%) <= 90`` AND ``Percentage (%) >= 4``.
    Rows whose percentage cells aren't parseable are skipped.
    """
    out = []
    for row in rows:
        pct = _parse_float(row.get("Percentage (%)"))
        cum = _parse_float(row.get("Cumulative Percentage (%)"))
        if pct is None or cum is None:
            continue
        if cum <= _SIGNIFICANT_CUM_PCT_MAX and pct >= _SIGNIFICANT_PCT_MIN:
            out.append(row)
    return out


def _parse_float(value):
    """Parse a CSV cell into ``float`` or ``None`` (handles empty / NaN strings)."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def _first_load_capture_event_set(capture_folder):
    """Open the first ``*.json.gz`` in ``capture_folder`` that loads cleanly.

    Returns ``traceEvents`` (list) or ``None`` if no file could be loaded.
    Files starting with ``execution_details`` are skipped (they're metadata).
    """
    if not capture_folder or not os.path.isdir(capture_folder):
        return None
    candidates = sorted(
        f
        for f in os.listdir(capture_folder)
        if f.endswith(".json.gz") and not f.startswith("execution_details")
    )
    for fname in candidates:
        try:
            data = _load_trace_json(os.path.join(capture_folder, fname))
        except (OSError, json.JSONDecodeError):
            continue
        return _events_of(data)
    return None


def _violator_evidence(label, rows, name_field="name", limit=3):
    """Build an evidence string listing the top-N violator op names."""
    names = []
    for row in rows[:limit]:
        n = (row.get(name_field) or "").strip()
        if n:
            names.append(n)
    head = f"{len(rows)} {label}"
    if names:
        return f"{head} (e.g. {', '.join(names)})"
    return head


# ---------------------------------------------------------------------------
# Profiling checks
# ---------------------------------------------------------------------------


def check_trace_missing(run_dir, _stream_file):
    # Step 1: trace path from cmd_prefix / category_manifest
    trace_path = resolve_trace_path(run_dir)
    if trace_path:
        if os.path.exists(trace_path):
            return None
        return FindingDraft(
            "Trace files missing",
            f"Trace file resolved but does not exist on disk: {trace_path}",
            "Check profiling setting in workload, or if trace path passed in correctly",
        )

    # Helper: list any *json.gz traces sitting 6 levels above the run_dir.
    def _nearby_traces():
        nearby_root = os.path.normpath(os.path.join(run_dir, *([".."] * 6)))
        if not os.path.isdir(nearby_root):
            return []
        try:
            return sorted(glob.glob(os.path.join(nearby_root, "*.json.gz")))[:10]
        except OSError:
            return []

    def _with_nearby(evidence):
        nearby = _nearby_traces()
        if nearby:
            return evidence + " | nearby *.json.gz: " + ", ".join(nearby)
        return evidence

    # Recursive scan: find every *.json.gz under run_dir/../../../../../../ (6 levels up).
    # Used only when no trace can be resolved at all, to help operators locate
    # the trace artefact wherever it actually landed.
    def _recursive_traces(limit=50):
        nearby_root = os.path.normpath(os.path.join(run_dir, *([".."] * 6)))
        if not os.path.isdir(nearby_root):
            return None, []
        found = []
        try:
            for dirpath, _, filenames in os.walk(nearby_root, followlinks=False):
                for name in filenames:
                    if name.endswith(".json.gz"):
                        found.append(os.path.join(dirpath, name))
                        if len(found) >= limit:
                            return nearby_root, sorted(found)
        except OSError:
            pass
        return nearby_root, sorted(found)

    # Step 2: trace_split exists with *.gz files but orchestrator never recorded a trace path
    split_dir = resolve_split_trace_dir(run_dir)
    if split_dir:
        gz_files = glob.glob(os.path.join(split_dir, "*.gz"))
        if gz_files:
            return FindingDraft(
                "Trace files missing (warning: trace_split present)",
                _with_nearby(
                    f"trace_split/ contains {len(gz_files)} *.gz file(s) but orchestrator did not receive a trace path "
                    f"(cmd_prefix.txt / category_manifest.json missing trace_path): {split_dir}"
                ),
                "Verify orchestrator wiring writes --profile_json_path / category_manifest.trace_path",
            )

    # Step 3: main rank-0 trace from trace_input_manifest
    main_trace = resolve_main_trace_path(run_dir)
    if main_trace:
        return FindingDraft(
            "Trace files missing",
            _with_nearby(
                f"Main trace exists ({main_trace}) but trace_split is missing and was not passed to orchestrator"
            ),
            "Re-run trace splitting and ensure split outputs are wired into the orchestrator manifest",
        )

    nearby_root, found = _recursive_traces()
    if nearby_root is None:
        listing = "<root not present>"
    elif not found:
        listing = f"<none under {nearby_root}>"
    else:
        listing = f"under {nearby_root}: " + ", ".join(found)

    return FindingDraft(
        "Trace files missing",
        f"trace_input_manifest.json returned empty / no main trace resolvable | recursive *.json.gz {listing}",
        "Verify trace_input_manifest.json is generated and points to a valid trace_input directory",
    )


def check_trace_size(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None
    size = os.path.getsize(trace_path)
    if size < _MIN_TRACE_BYTES:
        return FindingDraft(
            "Trace files too small (< 100KB)",
            f"Trace file is {size:,} bytes — may be empty or warmup-only",
            "Profiling window too short or no GPU ops captured",
        )
    if size > _MAX_TRACE_BYTES:
        return FindingDraft(
            "Trace files too large (> 5GB)",
            f"Trace file is {size / 1e9:.1f} GB — consider splitting",
            "Too many steps are being analyzed; reduce profiling window",
        )
    return None


def check_no_gpu_kernels(run_dir, _stream_file):
    events = _load_events(resolve_trace_path(run_dir))
    if events is None:
        return None
    if not any(e.get("cat") == "kernel" for e in events):
        return FindingDraft(
            "No GPU kernel events in trace",
            "Trace contains no events with cat='kernel'",
            "Ensure ProfilerActivity.CUDA is enabled in profiler config",
        )


def check_capture_missing(run_dir, _stream_file):
    capture_folder = resolve_capture_folder(run_dir)
    if capture_folder:
        if glob.glob(os.path.join(capture_folder, "*json.gz")):
            return None
        return FindingDraft(
            "Graph capture traces missing",
            f"Capture folder missing or empty: {capture_folder}",
            "Collect graph capture traces following Inference_analysis.md instructions",
        )

    manifest = _load_manifest(run_dir)
    if not manifest:
        return None
    capture_folder = manifest.get("capture_folder_path")
    if capture_folder and (
        not os.path.isdir(capture_folder) or not os.listdir(capture_folder)
    ):
        return FindingDraft(
            "Graph capture traces missing",
            f"Capture folder missing or empty: {capture_folder}",
            "Collect graph capture traces following Inference_analysis.md instructions",
        )
    return None


# ---------------------------------------------------------------------------
# Trace quality checks
# ---------------------------------------------------------------------------


def check_high_idle(run_dir, stream_file):
    timeline = _load_gpu_timeline(run_dir)
    idle = None
    source = None
    if timeline is not None and "idle_time" in timeline:
        idle = timeline["idle_time"]
        source = "perf_report_csvs/gpu_timeline.csv"
    else:
        manifest = _load_manifest(run_dir)
        if manifest:
            idle = manifest.get("gpu_utilization", {}).get("idle_time_percent")
            if idle is not None:
                source = "category_manifest.json gpu_utilization.idle_time_percent"

    if idle is not None and idle > _HIGH_IDLE_PCT:
        return FindingDraft(
            "High idle time (> 15% GPU idle)",
            f"GPU idle time is {idle:.1f}% ({source})",
            "Investigate CPU bottlenecks, graph capture enablement, host-side overhead",
        )
    return None


def check_gpu_graph_replay(run_dir, stream_file):
    if stream_contains(stream_file, r"DIAG:trace_quality:GPU_GRAPH_REPLAY"):
        return FindingDraft(
            "GPU Graph Replay detected in default mode",
            "DIAG tag found in agent stream",
            "Switch to inference analysis mode (--analysis_mode inference)",
        )
    # ToDo: add detailed trace-scan check once we have a stream file
    return None


def check_corrupt_json(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None
    if os.path.getsize(trace_path) > _MAX_TRACE_BYTES:
        return None

    try:
        _load_trace_json(trace_path)
    except ValueError as e:
        # orjson.JSONDecodeError subclasses ValueError; DataLoader also raises a
        # bare ValueError for an unknown/unloadable file type. Both mean the trace
        # is not parseable, so report corrupt rather than crash.
        return FindingDraft(
            "Trace file is corrupted/invalid JSON",
            f"{type(e).__name__}: {str(e)[:150]}",
            "Re-collect the trace; verify .json.gz files are not truncated",
        )
    except OSError as e:
        return FindingDraft(
            "Trace file is corrupted/invalid JSON",
            f"OSError reading trace: {str(e)[:150]}",
            "Re-collect the trace; verify .json.gz files are not truncated",
        )


def check_missing_cpu_op_shapes(run_dir, _stream_file):
    capture_folder = resolve_capture_folder(run_dir)
    if not capture_folder:
        return None
    events = _first_load_capture_event_set(capture_folder)
    if events is None:
        return None
    cpu_ops = [e for e in events if e.get("cat") == "cpu_op"]
    if not cpu_ops:
        return FindingDraft(
            "Trace missing cpu_op events with input shapes",
            f"Capture file has zero cpu_op events ({capture_folder})",
            "Profile with cpu_callstack and record_shapes enabled in profiler config",
        )
    with_shapes = sum(1 for e in cpu_ops if "Input Dims" in (e.get("args") or {}))
    if with_shapes < _MIN_CPU_OP_SHAPES:
        return FindingDraft(
            "Trace missing cpu_op events with input shapes",
            f"Only {with_shapes} of {len(cpu_ops)} cpu_op events carry 'Input Dims' in capture trace",
            "Profile with cpu_callstack and record_shapes enabled in profiler config",
        )
    return None


def check_inference_annotation_missing(run_dir, _stream_file):
    main_path = resolve_main_trace_path(run_dir)
    if not main_path or not os.path.exists(main_path):
        return FindingDraft(
            "Main inference trace missing ",
            f"No trace found at path: {main_path}",
            "Verify the correct trace was selected and that the inference mode was enabled",
        )
    if os.path.getsize(main_path) > _MAX_TRACE_BYTES:
        return None
    try:
        data = _load_trace_json(main_path)
    except (OSError, json.JSONDecodeError):
        # Narrow on purpose: main_path is glob-restricted to a .json* file, so the
        # bare ValueError("Unknown file type") that check_corrupt_json guards against
        # (arbitrary resolved path) cannot arise here.
        return FindingDraft(
            f"Main inference trace corrupted/invalid JSON at path: {main_path}",
            f"JSONDecodeError reading trace at path: {main_path}",
            "Verify the correct trace was selected and that the inference mode was enabled",
        )
    events = _events_of(data)
    execs = sum(
        1
        for e in events
        if e.get("cat") == "user_annotation"
        and isinstance(e.get("name"), str)
        and e["name"].startswith("execute_")
    )
    if execs == 0:
        return FindingDraft(
            "Main inference trace missing annotations in inference mode",
            f"Zero user_annotation events with name starting 'execute_' in {os.path.basename(main_path)}",
            "Profile again with the correct patches and inference profiler configs",
        )
    return None


def check_split_trace_missing(run_dir, _stream_file):
    split_dir = resolve_split_trace_dir(run_dir)
    if not split_dir:
        return FindingDraft(
            "Trace splits missing after initial split step",
            f"trace_split/ directory does not exist under {run_dir}",
            "Verify the correct trace was selected and that the split step was executed",
        )
    matches = glob.glob(os.path.join(split_dir, "mixed_steady_state*.json.gz"))
    if not matches:
        return FindingDraft(
            "Trace splits missing after initial split step",
            f"trace_split/ has no mixed_steady_state*.json.gz files in {split_dir}",
            "Verify the correct trace was selected and that the split step was executed",
        )
    return None


def check_split_incorrect(_run_dir, _stream_file):
    # Reserved placeholder for sublabel 2d; kept to preserve sublabel numbering.
    return None


def check_shape_profiler_missing(run_dir, _stream_file):
    capture_folder = resolve_capture_folder(run_dir)
    events = None
    source_name = None
    if capture_folder:
        events = _first_load_capture_event_set(capture_folder)
        source_name = capture_folder
    if events is None:
        trace_path = resolve_trace_path(run_dir)
        events = _load_events(trace_path)
        if events is None:
            return None
        source_name = os.path.basename(trace_path)

    sglang_cpu = sum(
        1
        for e in events
        if e.get("cat") == "cpu_op" and "sglang" in (e.get("name") or "").lower()
    )
    if sglang_cpu == 0:
        return None

    ksp = sum(
        1
        for e in events
        if e.get("cat") == "python_function"
        and "kernel_shape_profiler" in (e.get("name") or "")
    )
    if ksp == 0:
        return FindingDraft(
            "Trace was profiled without SGLang patches",
            f"sglang trace ({sglang_cpu} cpu_op matches) but no python_function event "
            f"contains 'kernel_shape_profiler' in {source_name}",
            "Re-profile with SGLang patches",
        )
    return None


def check_split_low_gpu_kernels(run_dir, _stream_file):
    trace_path = resolve_trace_path(run_dir)
    if not trace_path or not os.path.exists(trace_path):
        return None

    candidates = [trace_path]
    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        for pattern in ("decode_only*.json.gz", "prefilldecode*.json.gz"):
            for sibling in sorted(glob.glob(os.path.join(trace_dir, pattern))):
                if sibling not in candidates:
                    candidates.append(sibling)

    failures = []
    for path in candidates:
        events = _load_events(path)
        if events is None:
            continue
        cpu_ops = sum(1 for e in events if e.get("cat") == "cpu_op")
        kernels = sum(1 for e in events if e.get("cat") == "kernel")
        if cpu_ops == 0:
            continue
        ratio = kernels / cpu_ops
        if ratio < _SPLIT_LOW_KERNEL_RATIO:
            failures.append((path, ratio, kernels, cpu_ops))

    if not failures:
        return None

    parts = [
        f"{os.path.basename(p)}: kernel/cpu_op = {r:.2f} ({k} kernels, {c} cpu_ops)"
        for p, r, k, c in failures
    ]
    return FindingDraft(
        "No / very few GPU kernel events in split trace",
        f"{len(failures)} split trace(s) with kernel/cpu_op < {_SPLIT_LOW_KERNEL_RATIO}: "
        + "; ".join(parts),
        "Likely a spurious ROCm bug; apply correct fixes inside the docker container",
    )


def check_runtime_instability(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    unstable = []
    for row in rows:
        mean = _parse_float(row.get("Kernel Time (µs)_mean"))
        std = _parse_float(row.get("Kernel Time (µs)_std"))
        if not mean or std is None or mean <= 0:
            continue
        cv = std / mean
        if cv > _RUNTIME_INSTABILITY_CV:
            unstable.append((cv, row))
    if not unstable:
        return None
    unstable.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in unstable[:3]]
    names = ", ".join((r.get("name") or "")[:60] for r in top)
    return FindingDraft(
        "Run-time instability across iterations / per cpu_op",
        f"{len(unstable)} ops have Kernel Time std/mean > 0.25 (e.g. {names})",
        "Likely a profiler or system issue; verify if the variance is expected",
    )


# ---------------------------------------------------------------------------
# TraceLens agent workflow checks
# ---------------------------------------------------------------------------


def check_step1_fail(run_dir, stream_file):
    csv_dir = os.path.join(run_dir, "perf_report_csvs")
    if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
        return FindingDraft(
            "TraceLens perf reports are missing",
            f"perf_report_csvs/ missing or empty at {csv_dir}",
            "Check if the perf report generation command is correct",
        )
    required = ["gpu_timeline.csv", "unified_perf_summary.csv"]
    missing = [f for f in required if not os.path.isfile(os.path.join(csv_dir, f))]
    if missing:
        return FindingDraft(
            "TraceLens perf reports are missing",
            f"perf_report_csvs/ exists but is missing required files: {', '.join(missing)}",
            "Check if the perf report generation command is correct",
        )
    if stream_contains(
        stream_file,
        r"DIAG:pipeline:STEP1_FAIL|DIAG:tracelens_agent_workflow:PERF_REPORT_FAILURE",
    ):
        return FindingDraft(
            "TraceLens perf reports are missing",
            "DIAG tag found in agent stream",
            "Check if the perf report generation command is correct",
        )
    return None


def check_step2_5_fail(run_dir, stream_file):
    manifest_path = os.path.join(run_dir, "category_data", "category_manifest.json")
    if os.path.isdir(os.path.join(run_dir, "perf_report_csvs")) and not os.path.isfile(
        manifest_path
    ):
        return FindingDraft(
            "Orchestrator preparation fails (Steps 2-5)",
            "category_manifest.json missing despite perf_report_csvs/ existing",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )
    if stream_contains(
        stream_file,
        r"DIAG:pipeline:STEP2_5_FAIL|DIAG:tracelens_agent_workflow:ORCHESTRATOR_PREP_FAIL",
    ):
        return FindingDraft(
            "Orchestrator preparation fails (Steps 2-5)",
            "DIAG tag found in agent stream",
            "Check that perf_report_csvs/ was generated in Step 1; verify trace path and platform",
        )
    return None


def check_output_incomplete(run_dir, _stream_file):
    expected = ["metadata", "category_data", "system_findings", "category_findings"]
    missing = [d for d in expected if not os.path.isdir(os.path.join(run_dir, d))]
    if missing:
        return FindingDraft(
            "Output directory structure incomplete",
            f"Missing directories: {', '.join(missing)}",
            "Re-run analysis pipeline from Step 1; check disk space and write permissions",
        )
    return None


def check_prefix_fail(_run_dir, stream_file):
    if stream_contains(
        stream_file,
        r"DIAG:pipeline:PREFIX_FAIL|DIAG:tracelens_agent_workflow:CMD_PREFIX_INVALID",
    ):
        return FindingDraft(
            "Command prefix validation fails",
            "DIAG tag found in agent stream",
            "Verify TraceLens is installed; check SSH/container/venv connectivity",
        )
    return None


# ---------------------------------------------------------------------------
# TraceLens agent workflow checks (continued — report)
# ---------------------------------------------------------------------------


def check_report_too_small(run_dir, _stream_file):
    report = os.path.join(run_dir, "analysis.md")
    if os.path.isfile(report) and os.path.getsize(report) < 100:
        return FindingDraft(
            "analysis.md too small (< 100 bytes)",
            f"analysis.md is {os.path.getsize(report)} bytes",
            "Report generation likely crashed mid-write",
        )
    if not os.path.isfile(report) and os.path.isdir(
        os.path.join(run_dir, "category_findings")
    ):
        return FindingDraft(
            "analysis.md missing",
            "analysis.md does not exist but category_findings/ is present",
            "Report generation likely crashed or was interrupted",
        )
    return None


# ---------------------------------------------------------------------------
# TraceLens agent workflow checks (continued — subagent) / infra
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
            if rtype == "system" and "RESOURCE_EXHAUSTED" in str(
                record.get("error", "")
            ):
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
    return None


def check_subagent_budget(run_dir, _stream_file):
    manifest = _load_manifest(run_dir)
    if not manifest:
        return None
    findings_dir = os.path.join(run_dir, "category_findings")
    system_dir = os.path.join(run_dir, "system_findings")

    idle_pct = manifest.get("gpu_utilization", {}).get("idle_time_percent")

    missing = []
    for cat in manifest.get("categories", []):
        name = cat.get("name", "")
        if not name:
            continue
        if name == "cpu_idle" and idle_pct is not None and idle_pct <= _HIGH_IDLE_PCT:
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
    return None


# ---------------------------------------------------------------------------
# Infrastructure checks
# ---------------------------------------------------------------------------


def check_ssh_fail(_run_dir, stream_file):
    match = stream_find(
        stream_file,
        r"(ssh:.*(?:timeout|timed out)|Connection refused|Permission denied \(publickey)",
    )
    if match:
        return FindingDraft(
            "SSH connection to node fails",
            match,
            "Verify node hostname, SSH keys, and network connectivity",
        )
    return None


def check_docker_missing(_run_dir, stream_file):
    match = stream_find(
        stream_file,
        r"(Error: No such container|Error response from daemon:.*not running)",
    )
    if match:
        return FindingDraft(
            "Docker container not found",
            match,
            "Start the container or verify container name; check docker ps",
        )
    return None


def check_tl_not_installed(_run_dir, stream_file):
    match = stream_find(
        stream_file,
        r"(ModuleNotFoundError: No module named 'TraceLens'|No module named 'TraceLens')",
    )
    if match:
        return FindingDraft(
            "TraceLens not installed in remote env",
            match,
            "Run pip install git+https://github.com/AMD-AGI/TraceLens.git",
        )
    return None


def check_disk_full(run_dir, stream_file):
    zero_byte = []
    for dirpath, _dirnames, filenames in os.walk(run_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if os.path.getsize(fpath) == 0 and not fname.startswith("."):
                zero_byte.append(os.path.relpath(fpath, run_dir))
    if len(zero_byte) > _MAX_ZERO_BYTE_FILES:
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
    return None


def check_nfs_stale(_run_dir, stream_file):
    match = stream_find(stream_file, r"(Stale file handle|stale NFS|NFS.*stale)")
    if match:
        return FindingDraft(
            "NFS latency / stale file handles",
            match,
            "Add small delays between write and read operations; verify NFS mount",
        )
    return None


def check_missing_dep(_run_dir, stream_file):
    match = stream_find(
        stream_file,
        r"(ImportError: No module named|ModuleNotFoundError: No module named)(?!.*TraceLens)",
    )
    if match:
        return FindingDraft(
            "Python dependency missing",
            match,
            "Install TraceLens with all dependencies",
        )
    return None


# ---------------------------------------------------------------------------
# GEAK interface checks
# ---------------------------------------------------------------------------


def check_kernel_candidates_missing(run_dir_or_session_dir, _stream_file=None):
    """Check if kernel_candidates.json exists for a session that reached the KERNEL phase.

    Accepts either a Hyperloom session directory (containing
    ``session_breakdown.json``) or a TraceLens run_dir nested inside one
    (``<session>/kernel-agent/runs/<id>/<tl-hash>/tracelens/``).
    Returns None when the file exists, session_breakdown.json is absent,
    or the session never reached KERNEL.
    """
    path = os.path.abspath(run_dir_or_session_dir)

    if os.path.isfile(os.path.join(path, "session_breakdown.json")):
        session_dir = path
    else:
        session_dir = os.path.normpath(os.path.join(path, *([".."] * 5)))

    sbd_path = os.path.join(session_dir, "session_breakdown.json")
    if not os.path.isfile(sbd_path):
        return None

    try:
        with open(sbd_path) as f:
            sbd = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Hyperloom writes the kernel phase as the canonical "KERNEL_AGENT"
    phases = [p.get("phase", "") for p in sbd.get("phase_segments", [])]
    if "KERNEL_AGENT" not in phases:
        return None

    matches = glob.glob(
        os.path.join(
            session_dir,
            "kernel-agent",
            "runs",
            "*",
            "*",
            "kernel_candidates.json",
        )
    )
    if matches:
        return None

    return FindingDraft(
        "kernel_candidates.json missing",
        f"Session reached KERNEL phase but no kernel_candidates.json found "
        f"under {session_dir}/kernel-agent/runs/",
        "TraceLens analysis did not produce kernel_candidates.json; "
        "check tl-* run logs under kernel-agent/runs/",
    )


# ---------------------------------------------------------------------------
# Perf model / report-correctness checks
# ---------------------------------------------------------------------------


def check_perf_report_command_incorrect(_run_dir, _stream_file):
    return None


def check_hipgraph_launch_in_report(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    matches = [r for r in rows if "GraphLaunch" in (r.get("name") or "")]
    if not matches:
        return None
    return FindingDraft(
        "HipGraph/cuGraph launch events appear in the perf report",
        _violator_evidence(
            "rows reference *GraphLaunch* in unified_perf_summary.csv", matches
        ),
        "Investigate perf report generation; verify command-graph capture traces are present "
        "and that graph + capture merge passed",
    )


def check_capture_graph_merge_fail(_run_dir, _stream_file):
    return None


def check_synthetic_op_significant(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    sig = _significant_rows(rows)
    matches = [r for r in sig if "(Synthetic Op)" in (r.get("name") or "")]
    if not matches:
        return None
    return FindingDraft(
        "Synthetic op appears among significant ops",
        _violator_evidence("significant ops contain '(Synthetic Op)'", matches),
        "Possibly profiler shape/dtype capture was not enabled; perf model may be missing. "
        "Add perf model — TraceLens side fix",
    )


def check_unclassified_op_significant(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    sig = _significant_rows(rows)
    matches = [
        r for r in sig if (r.get("op category") or "").strip().lower() in {"", "other"}
    ]
    if not matches:
        return None
    return FindingDraft(
        "Unclassified op among significant ops",
        _violator_evidence("significant ops have empty / 'other' op category", matches),
        "Add perf model — TraceLens side fix",
    )


def check_tbs_tflops_missing(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    matches = []
    for row in _significant_rows(rows):
        cat = (row.get("op category") or "").lower()
        if any(s in cat for s in _TBS_TFLOPS_EXEMPT_SUBSTRINGS):
            continue
        tbs = _parse_float(row.get("TB/s_mean"))
        tflops = _parse_float(row.get("TFLOPS/s_mean"))
        if tbs is None and tflops is None:
            matches.append(row)
    if not matches:
        return None
    return FindingDraft(
        "TBs / TFLOPs not recorded for significant ops",
        _violator_evidence(
            "significant ops missing both TB/s_mean and TFLOPS/s_mean", matches
        ),
        "Likely missing shapes; add/debug perf model — TraceLens side fix",
    )


def check_roofline_pct_missing(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    matches = []
    for row in _significant_rows(rows):
        tbs = _parse_float(row.get("TB/s_mean"))
        tflops = _parse_float(row.get("TFLOPS/s_mean"))
        roofline = _parse_float(row.get("Pct Roofline_mean"))
        if (tbs is not None or tflops is not None) and roofline is None:
            matches.append(row)
    if not matches:
        return None
    return FindingDraft(
        "pct of roofline missing",
        _violator_evidence(
            "significant ops have TB/s or TFLOPs but no Pct Roofline_mean", matches
        ),
        "Likely missing compute precision metadata; TraceLens side fix",
    )


def check_zero_pct_ops(run_dir, _stream_file):
    rows = _load_unified_perf_summary(run_dir)
    if not rows:
        return None
    matches = []
    for row in rows:
        pct = _parse_float(row.get("Percentage (%)"))
        if pct is not None and pct == 0:
            matches.append(row)
    if not matches:
        return None
    return FindingDraft(
        "Op with zero recorded percentage (truncated at < 10us kernel time)",
        _violator_evidence(
            "rows have Percentage (%) == 0 in unified_perf_summary.csv", matches
        ),
        "No GPU kernel recorded for these ops; perf model failed for that specific instance "
        "— TraceLens side fix",
    )


# ---------------------------------------------------------------------------
# Check registry
#
# Every check has an ordinal sublabel (e.g. "1a", "2f", "4b") so the DIAG
# tag is always DIAG:{category}:{sublabel}_{NAME}.
#
# Section numbering:
#   1x = profiling          (data capture / workload side)
#   2x = trace_quality      (trace / runtime quality)
#   3x = perf_model         (report correctness / model coverage)
#   4x = tracelens_agent_workflow  (orchestrator + agent execution)
#   5x = infra              (host / environment)
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    # -- 1x: profiling ------------------------------------------------------
    CheckSpec(
        "1a",
        "TRACE_MISSING",
        "profiling",
        check_trace_missing,
    ),
    CheckSpec("1b", "TRACE_SIZE", "profiling", check_trace_size),
    CheckSpec("1c", "NO_GPU_KERNELS", "profiling", check_no_gpu_kernels),
    CheckSpec("1d", "CAPTURE_MISSING", "profiling", check_capture_missing),
    # -- 2x: trace_quality --------------------------------------------------
    CheckSpec(
        "2a",
        "MISSING_CPU_OP_SHAPES",
        "trace_quality",
        check_missing_cpu_op_shapes,
        detailed_only=True,
    ),
    CheckSpec(
        "2b",
        "INFERENCE_ANNOTATION_MISSING",
        "trace_quality",
        check_inference_annotation_missing,
        detailed_only=True,
    ),
    CheckSpec(
        "2c",
        "SPLIT_TRACE_MISSING",
        "trace_quality",
        check_split_trace_missing,
        detailed_only=True,
        implies_failures=["2f"],
    ),
    CheckSpec(
        "2d",
        "SPLIT_INCORRECT",
        "trace_quality",
        check_split_incorrect,
        detailed_only=True,
    ),
    CheckSpec(
        "2e",
        "SHAPE_PROFILER_MISSING",
        "trace_quality",
        check_shape_profiler_missing,
        detailed_only=True,
    ),
    CheckSpec(
        "2f",
        "SPLIT_LOW_GPU_KERNELS",
        "trace_quality",
        check_split_low_gpu_kernels,
        detailed_only=True,
    ),
    CheckSpec("2g", "HIGH_IDLE", "trace_quality", check_high_idle),
    CheckSpec("2h", "GPU_GRAPH_REPLAY", "trace_quality", check_gpu_graph_replay),
    CheckSpec("2i", "CORRUPT_JSON", "trace_quality", check_corrupt_json),
    CheckSpec("2j", "RUNTIME_INSTABILITY", "trace_quality", check_runtime_instability),
    # -- 3x: perf_model -----------------------------------------------------
    CheckSpec(
        "3a",
        "PERF_REPORT_COMMAND_INCORRECT",
        "perf_model",
        check_perf_report_command_incorrect,
    ),
    CheckSpec(
        "3b", "HIPGRAPH_LAUNCH_IN_REPORT", "perf_model", check_hipgraph_launch_in_report
    ),
    CheckSpec(
        "3c", "CAPTURE_GRAPH_MERGE_FAIL", "perf_model", check_capture_graph_merge_fail
    ),
    CheckSpec(
        "3d", "SYNTHETIC_OP_SIGNIFICANT", "perf_model", check_synthetic_op_significant
    ),
    CheckSpec(
        "3e",
        "UNCLASSIFIED_OP_SIGNIFICANT",
        "perf_model",
        check_unclassified_op_significant,
    ),
    CheckSpec("3f", "TBS_TFLOPS_MISSING", "perf_model", check_tbs_tflops_missing),
    CheckSpec("3g", "ROOFLINE_PCT_MISSING", "perf_model", check_roofline_pct_missing),
    CheckSpec("3h", "ZERO_PCT_OPS", "perf_model", check_zero_pct_ops),
    # -- 4x: tracelens_agent_workflow ----------------------------------------
    CheckSpec(
        "4a",
        "PERF_REPORT_FAILURE",
        "tracelens_agent_workflow",
        check_step1_fail,
        implies_failures=["3b", "3d", "3e", "3f", "3g", "3h"],
    ),
    CheckSpec(
        "4b", "ORCHESTRATOR_PREP_FAIL", "tracelens_agent_workflow", check_step2_5_fail
    ),
    CheckSpec(
        "4c", "OUTPUT_DIRS_MISSING", "tracelens_agent_workflow", check_output_incomplete
    ),
    CheckSpec(
        "4d", "CMD_PREFIX_INVALID", "tracelens_agent_workflow", check_prefix_fail
    ),
    CheckSpec(
        "4e",
        "ANALYSIS_MD_MISSING_OR_EMPTY",
        "tracelens_agent_workflow",
        check_report_too_small,
    ),
    CheckSpec(
        "4f",
        "SUBAGENT_FINDINGS_MISSING",
        "tracelens_agent_workflow",
        check_subagent_budget,
    ),
    # -- 5x: infra -----------------------------------------------------------
    CheckSpec("5a", "SSH_FAIL", "infra", check_ssh_fail),
    CheckSpec("5b", "DOCKER_MISSING", "infra", check_docker_missing),
    CheckSpec("5c", "TL_NOT_INSTALLED", "infra", check_tl_not_installed),
    CheckSpec("5d", "DISK_FULL", "infra", check_disk_full),
    CheckSpec("5e", "NFS_STALE", "infra", check_nfs_stale),
    CheckSpec("5f", "MISSING_DEP", "infra", check_missing_dep),
    CheckSpec("5g", "CONTEXT_LENGTH_EXCEEDED", "infra", check_resource_exhausted),
    # -- 6x: geak_interface ---------------------------------------------------
    CheckSpec(
        "6a",
        "KERNEL_CANDIDATES_MISSING",
        "geak_interface",
        check_kernel_candidates_missing,
    ),
]
