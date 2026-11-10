###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Profile checks on a Magpie benchmark directory, before any analysis exists.

``triage.runner --bench-dir`` runs them through ``PROFILE_CHECKS`` in ``checks.py``.
They look at rank 0 only and stream each trace once instead of loading the JSON.
Requirements are listed in ``TraceLens/Agent/Profiling/trace_collection_requirements.md``.
Speculative decoding and AgentX are read from ``config.yaml`` next to ``torch_trace``.
"""

from __future__ import annotations

import functools
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from TraceLens.TraceUtils.utils.annotation_utils import (
    ATOM_DETAILED_PATTERN,
    ATOM_NATIVE_PATTERN,
    SGLANG_DETAILED_PATTERN,
    SGLANG_NATIVE_PATTERN,
    VLLM_DETAILED_PATTERN,
    VLLM_NATIVE_PATTERN,
)
from TraceLens.TraceUtils.utils.detect_utils import DetectStatus

# Constants

EXPECTED_STEPS = 256
READ_CHUNK = 8 << 20
# cpu_op Input Dims sit in the args object just after the name. The overlap
# has to cover that window so a match split across two reads is counted once.
EVENT_WINDOW = 4096

CAPTURE_DIR_NAMES = {"capture_traces", "graph_capture_profile"}
SKIP_DIR_NAMES = CAPTURE_DIR_NAMES | {"trace_split"}
TRACE_SUFFIXES = (".json.gz", ".json", ".pt.trace.json.gz")

DRAFTER_FILE_RE = re.compile(r"(speculator|draft|eagle)", re.IGNORECASE)
STAGE_FILE_RE = re.compile(r"-(DECODE|EXTEND)\.trace\.json\.gz$", re.IGNORECASE)
RANK0_TP_RE = re.compile(r"(^|[^A-Za-z0-9])TP-0([^0-9]|$)")
RANK0_NAME_RE = re.compile(r"rank[-_]?0([^0-9]|$)", re.IGNORECASE)
FRAMEWORK_RE = re.compile(r"^framework:\s*(\w+)\s*$", re.MULTILINE)
SPEC_OVERRIDE_RE = re.compile(r"SPEC_NUM_TOKENS_OVERRIDE:\s*['\"]?(\d+)")
SPEC_FLAG_RES = (
    re.compile(r"--speculative-algorithm\b", re.IGNORECASE),
    re.compile(r"--speculative-draft", re.IGNORECASE),
    re.compile(r"--spec-method\b", re.IGNORECASE),
    re.compile(r"--method\s+eagle", re.IGNORECASE),
    re.compile(r"--draft-model\b", re.IGNORECASE),
    re.compile(r"--num-speculative-tokens\s+([1-9]\d*)", re.IGNORECASE),
    re.compile(r"speculative-num-draft-tokens\s+([1-9]\d*)", re.IGNORECASE),
)
# SGLang eagle/MTP builds emit VERIFY and DRAFT with the same sqsq aggregates
# as EXTEND/DECODE/MIXED. TraceUtils does not list those phases yet.
SGLANG_OTHER_DETAILED_RE = re.compile(
    r"step\[(?!EXTEND\b|DECODE\b|MIXED\b)\w+\b[^\]]*sqsq=\d+[^\]]*\]",
    re.IGNORECASE,
)
STEP_PREFIXES = ("execute_", "step[", "prefill[", "decode[")
PHASE_RE = re.compile(r"step\[(\w+)", re.IGNORECASE)

USER_ANN_RE = re.compile(
    rb'"cat": "user_annotation"\s*,\s*"name": "([^"]+)"',
)
CPU_OP_MARK = b'"cat": "cpu_op"'
PYTHON_FUNCTION_MARK = b'"cat": "python_function"'
KERNEL_MARK = b'"cat": "kernel"'
INPUT_DIMS_MARK = b'"Input Dims"'
NAME_RE = re.compile(rb'"name": "([^"]*)"')
GRAPH_LAUNCH_RE = re.compile(rb'"name": "(?:hip|cuda)GraphLaunch\w*"')
SHAPE_DIGIT_RE = re.compile(rb"\d")

BENCHMARK_REPORT = "benchmark_report.json"
ERROR_LINE_RE = re.compile(
    r"Error\b|Exception\b|Traceback|died|Killed|out of memory", re.IGNORECASE
)
XTRACE_PREFIX_RE = re.compile(r"^\++ (?:echo )?")
SPLIT_MANIFEST = "split_manifest.json"
MIXED_SPLIT_RE = re.compile(
    r"^mixed_steady_state_prefill_\d+_prefilldecode_(\d+)_decode_(\d+)_"
)


class Check:
    def __init__(self, name, status, detail):
        self.name = name
        self.status = status
        self.detail = detail


def resolve_torch_trace(path):
    """Return ``(torch_trace_dir, config_text)`` for a benchmark or trace dir."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_dir():
        raise SystemExit(f"not a directory: {candidate}")
    if candidate.name == "torch_trace":
        torch_trace = candidate
        config_path = candidate.parent / "config.yaml"
    elif (candidate / "torch_trace").is_dir():
        torch_trace = candidate / "torch_trace"
        config_path = candidate / "config.yaml"
    else:
        raise SystemExit(
            f"{candidate} is not a torch_trace directory and has no torch_trace/"
        )
    config_text = ""
    if config_path.is_file():
        config_text = config_path.read_text(encoding="utf-8", errors="replace")
    return torch_trace, config_text


def detect_framework(config_text, torch_trace):
    match = FRAMEWORK_RE.search(config_text)
    if match:
        return match.group(1).lower()
    names = " ".join(p.name for p in torch_trace.rglob("*") if p.is_file())
    if "TP-" in names or "graph_capture_profile" in names:
        return "sglang"
    if (torch_trace / "rank_0").is_dir():
        return "atom"
    return "vllm"


def detect_speculative(config_text):
    """True, False, or None when config.yaml does not say."""
    if not config_text:
        return None
    override = SPEC_OVERRIDE_RE.search(config_text)
    if override is not None and int(override.group(1)) == 0:
        return False
    if any(pattern.search(config_text) for pattern in SPEC_FLAG_RES):
        return True
    return False


def _is_capture_path(path):
    return any(part in CAPTURE_DIR_NAMES for part in path.parts)


def _is_skipped_path(path):
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _is_trace_file(path):
    name = path.name
    return name.endswith(TRACE_SUFFIXES) and path.is_file()


def capture_files(torch_trace):
    files = []
    for path in torch_trace.rglob("*"):
        if path.name == "execution_details.json":
            continue
        if path.is_file() and _is_capture_path(path) and _is_trace_file(path):
            if path.stat().st_size > 0:
                files.append(path)
    return files


def _is_rank0(path):
    if path.name.startswith("merged") or "async_llm" in path.name:
        return False
    if RANK0_TP_RE.search(path.name) or RANK0_NAME_RE.search(path.name):
        return True
    return path.parent.name in {"rank_0", "rank-0", "rank0"}


def main_traces(torch_trace):
    """Rank-0 serving traces, preferring the combined file over stage splits."""
    found = []
    for path in torch_trace.rglob("*"):
        if not _is_trace_file(path) or _is_skipped_path(path):
            continue
        if _is_rank0(path):
            found.append(path)
    combined = [path for path in found if not STAGE_FILE_RE.search(path.name)]
    chosen = combined or found
    return sorted(chosen)


def classify_annotation(name):
    if VLLM_DETAILED_PATTERN.match(name):
        return "vllm_detailed"
    if SGLANG_DETAILED_PATTERN.match(name):
        return "sglang_detailed"
    if ATOM_DETAILED_PATTERN.match(name):
        return "atom_detailed"
    if SGLANG_OTHER_DETAILED_RE.match(name):
        return "sglang_detailed_other"
    if VLLM_NATIVE_PATTERN.match(name):
        return "vllm_native"
    if SGLANG_NATIVE_PATTERN.match(name):
        return "sglang_native"
    if ATOM_NATIVE_PATTERN.match(name):
        return "atom_native"
    if name.startswith(STEP_PREFIXES):
        return "unparsed_step"
    return ""


def _open_trace(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def _has_shape(window):
    dims_at = window.find(INPUT_DIMS_MARK)
    if dims_at < 0:
        return False
    return SHAPE_DIGIT_RE.search(window[dims_at : dims_at + 500]) is not None


def scan_trace(path):
    """Stream a trace once and count the events the checks need.

    Each read keeps the last ``EVENT_WINDOW`` bytes for the next read, and
    only counts matches that start before that tail, so an event split across
    two reads is counted once.
    """
    kinds = Counter()
    phases = Counter()
    stats = Counter()
    truncated = False
    leftover = b""

    def consume(data, final):
        limit = len(data) if final else max(0, len(data) - EVENT_WINDOW)
        for match in USER_ANN_RE.finditer(data):
            if match.start() >= limit:
                break
            name = match.group(1).decode("utf-8", "replace")
            kind = classify_annotation(name)
            if not kind:
                continue
            kinds[kind] += 1
            phase = PHASE_RE.match(name)
            if phase:
                phases[phase.group(1).upper()] += 1
        for mark, key in (
            (PYTHON_FUNCTION_MARK, "python_function"),
            (KERNEL_MARK, "kernel"),
        ):
            position = 0
            while True:
                found = data.find(mark, position)
                if found < 0 or found >= limit:
                    break
                stats[key] += 1
                position = found + len(mark)
        for match in GRAPH_LAUNCH_RE.finditer(data):
            if match.start() >= limit:
                break
            stats["graph_launch"] += 1
        position = 0
        while True:
            found = data.find(CPU_OP_MARK, position)
            if found < 0 or found >= limit:
                break
            window = data[found : found + EVENT_WINDOW]
            name_match = NAME_RE.search(window)
            name = name_match.group(1) if name_match else b""
            has_shape = _has_shape(window)
            stats["cpu_op"] += 1
            if has_shape:
                stats["cpu_op_with_shape"] += 1
            if b"sglang_profiler::" in name:
                stats["sglang_profiler"] += 1
                if has_shape:
                    stats["sglang_profiler_with_shape"] += 1
            position = found + len(CPU_OP_MARK)
        return limit

    try:
        with _open_trace(path) as handle:
            while True:
                chunk = handle.read(READ_CHUNK)
                if not chunk:
                    consume(leftover, True)
                    break
                data = leftover + chunk
                limit = consume(data, False)
                leftover = data[limit:]
    except EOFError:
        truncated = True
        if leftover:
            consume(leftover, True)
    return kinds, phases, stats, truncated


def _step_total(kinds):
    return sum(kinds.values())


def _detailed_total(kinds):
    return sum(
        kinds[name]
        for name in (
            "vllm_detailed",
            "sglang_detailed",
            "sglang_detailed_other",
            "atom_detailed",
        )
    )


def _native_total(kinds):
    return sum(kinds[name] for name in ("vllm_native", "sglang_native", "atom_native"))


def check_trace(torch_trace, framework, speculative, expected_steps):
    checks = []
    captures = capture_files(torch_trace)
    drafter = [path for path in captures if DRAFTER_FILE_RE.search(path.name)]
    target = [path for path in captures if path not in drafter]
    capture_dirs = sorted(
        {path.parent.relative_to(torch_trace).as_posix() for path in captures}
    )
    where = ", ".join(capture_dirs) if capture_dirs else "no capture directory"
    if target:
        checks.append(
            Check(
                "capture traces",
                "PASS",
                f"{len(target)} target capture file(s) in {where}",
            )
        )
    else:
        checks.append(
            Check(
                "capture traces",
                "FAIL",
                f"no target capture traces under {torch_trace} "
                "(looked in capture_traces/ and graph_capture_profile/)",
            )
        )

    if speculative is True:
        if drafter:
            checks.append(
                Check(
                    "drafter capture traces",
                    "PASS",
                    f"{len(drafter)} drafter capture file(s)",
                )
            )
        else:
            checks.append(
                Check(
                    "drafter capture traces",
                    "FAIL",
                    "speculative decoding is on and no drafter capture traces "
                    "were found (expected speculator, EAGLEDraft, or draft_step files)",
                )
            )
    elif speculative is False:
        checks.append(
            Check(
                "drafter capture traces",
                "SKIP",
                "speculative decoding is off",
            )
        )
    else:
        status = "PASS" if drafter else "SKIP"
        detail = (
            f"{len(drafter)} drafter capture file(s); config.yaml did not say"
            if drafter
            else "config.yaml not found; drafter captures not required"
        )
        checks.append(Check("drafter capture traces", status, detail))

    traces = main_traces(torch_trace)
    if len(traces) != 1:
        detail = (
            "no rank-0 serving trace"
            if not traces
            else "expected one rank-0 capture, found "
            + ", ".join(path.name for path in traces)
        )
        checks.append(Check("single rank-0 trace", "FAIL", detail))
        for name in (
            "detailed annotations",
            "step count",
            "kernel shape profiler",
            "cpu call stack",
            "cpu_op shapes",
            "gpu kernels",
            "graph launches",
        ):
            checks.append(Check(name, "FAIL", "no single rank-0 trace to scan"))
        checks.append(check_benchmark_report(torch_trace.parent))
        return checks

    trace_path = traces[0]
    kinds, phases, stats, truncated = scan_trace(trace_path)
    if truncated:
        checks.append(
            Check(
                "trace readable",
                "FAIL",
                f"{trace_path.name} ended before the gzip stream finished",
            )
        )
    detailed = _detailed_total(kinds)
    native = _native_total(kinds)
    if detailed:
        phase_detail = ""
        if phases:
            phase_detail = "; " + ", ".join(
                f"{phase} {count}" for phase, count in sorted(phases.items())
            )
        checks.append(
            Check(
                "detailed annotations",
                "PASS",
                f"{detailed} detailed user annotation(s) in {trace_path.name}"
                + phase_detail,
            )
        )
    elif native:
        checks.append(
            Check(
                "detailed annotations",
                "FAIL",
                f"{trace_path.name} has {native} native annotation(s) and no "
                "sqsq/sqsk aggregates (detailed_trace_annotation / "
                "detailed_annotations / ATOM_ENABLE_DETAILED_ANNOTATION)",
            )
        )
    else:
        checks.append(
            Check(
                "detailed annotations",
                "FAIL",
                f"no iteration user annotations in {trace_path.name}",
            )
        )

    steps = _step_total(kinds)
    checks.append(
        Check(
            "step count",
            "PASS" if steps == expected_steps else "FAIL",
            f"{steps} step(s) in {trace_path.name}, expected {expected_steps}",
        )
    )

    if framework == "sglang":
        shaped = stats["sglang_profiler_with_shape"]
        seen = stats["sglang_profiler"]
        if shaped:
            checks.append(
                Check(
                    "kernel shape profiler",
                    "PASS",
                    f"{shaped} sglang_profiler cpu_op(s) with Input Dims",
                )
            )
        elif seen:
            checks.append(
                Check(
                    "kernel shape profiler",
                    "FAIL",
                    f"{seen} sglang_profiler cpu_op(s) but none have Input Dims",
                )
            )
        else:
            checks.append(
                Check(
                    "kernel shape profiler",
                    "FAIL",
                    "no sglang_profiler:: cpu_op events "
                    "(TRACELENS_SHAPE_DISCOVERY and the kernel shape tool)",
                )
            )
    else:
        checks.append(
            Check(
                "kernel shape profiler",
                "SKIP",
                f"not an SGLang trace ({framework})",
            )
        )

    if stats["python_function"]:
        checks.append(
            Check(
                "cpu call stack",
                "PASS",
                f"{stats['python_function']} python_function event(s)",
            )
        )
    else:
        checks.append(
            Check(
                "cpu call stack",
                "FAIL",
                "no python_function events (with_stack / ATOM_PROFILER_MORE)",
            )
        )

    if stats["cpu_op_with_shape"]:
        checks.append(
            Check(
                "cpu_op shapes",
                "PASS",
                f"{stats['cpu_op_with_shape']} of {stats['cpu_op']} cpu_op(s) "
                "have Input Dims",
            )
        )
    else:
        checks.append(
            Check(
                "cpu_op shapes",
                "FAIL",
                f"{stats['cpu_op']} cpu_op(s) and none have Input Dims "
                "(record_shapes / SGLANG_PROFILE_RECORD_SHAPES / ATOM_PROFILER_MORE)",
            )
        )

    checks.append(
        Check(
            "gpu kernels",
            "PASS" if stats["kernel"] else "FAIL",
            f"{stats['kernel']} GPU kernel event(s) in {trace_path.name}",
        )
    )

    launches = stats["graph_launch"]
    if not captures:
        checks.append(Check("graph launches", "SKIP", "no capture traces to merge"))
    else:
        checks.append(
            Check(
                "graph launches",
                "PASS" if launches else "FAIL",
                f"{launches} graph launch(es) in {trace_path.name}"
                + ("" if launches else "; capture traces have nothing to merge into"),
            )
        )
    checks.append(check_benchmark_report(torch_trace.parent))
    return checks


def check_benchmark_report(benchmark_dir):
    """Warn when Magpie marked the benchmark failed; the trace checks decide."""
    path = benchmark_dir / BENCHMARK_REPORT
    if not path.is_file():
        return Check("benchmark report", "SKIP", f"no {BENCHMARK_REPORT}")
    report = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if report.get("success"):
        throughput = report.get("throughput") or {}
        latency = report.get("latency") or {}
        return Check(
            "benchmark report",
            "PASS",
            f"output {throughput.get('output_throughput')} tok/s, "
            f"TTFT {(latency.get('ttft') or {}).get('median_ms')} ms, "
            f"TPOT {(latency.get('tpot') or {}).get('median_ms')} ms (median)",
        )
    return Check(
        "benchmark report",
        "WARN",
        f"marked failed: {_failure_reason(report.get('errors') or [])}",
    )


def _failure_reason(errors):
    """Last error-looking line across the recorded errors (stderr is verbose)."""
    lines = [
        line.strip()
        for error in errors
        for line in str(error).splitlines()
        if line.strip()
    ]
    flagged = [line for line in lines if ERROR_LINE_RE.search(line)]
    if flagged:
        return XTRACE_PREFIX_RE.sub("", flagged[-1]).strip("'\"")
    return lines[0] if lines else "no error recorded"


def expected_steps(config_text):
    """Steps the rank-0 trace should hold: AgentX ``num_steps``, else ``EXPECTED_STEPS``."""
    config = yaml.safe_load(config_text or "") or {}
    if (config.get("agentx") or {}).get("enabled"):
        profiler = (config.get("profiler") or {}).get("torch_profiler") or {}
        return profiler.get("num_steps", EXPECTED_STEPS)
    return EXPECTED_STEPS


def check_split(benchmark_dir):
    """The splitter returns normally when it refuses to split, so read its manifest."""
    manifests = sorted(benchmark_dir.rglob(SPLIT_MANIFEST))
    if not manifests:
        return [
            Check("split traces", "FAIL", f"no {SPLIT_MANIFEST} under {benchmark_dir}")
        ]
    split_dir = manifests[-1].parent
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    mixed = [MIXED_SPLIT_RE.match(path.name) for path in split_dir.iterdir()]
    mixed = [match for match in mixed if match]
    problems = []
    if manifest.get("aborted"):
        problems.append("split aborted")
    if manifest.get("status") != DetectStatus.SPLITTABLE:
        problems.append(f"status {manifest.get('status')}")
    if manifest.get("phase_confidence") != "high":
        problems.append(f"phase confidence {manifest.get('phase_confidence')}")
    if manifest.get("suspiciously_few_roots"):
        problems.append("suspiciously few roots")
    if any(int(match.group(1)) + int(match.group(2)) == 0 for match in mixed):
        problems.append("mixed split has no decode steps")
    return [
        Check(
            "split traces",
            "PASS" if mixed else "FAIL",
            f"{len(mixed)} mixed steady-state split(s) in {split_dir}",
        ),
        Check(
            "split quality",
            "FAIL" if problems else "PASS",
            f"{'; '.join(problems) or 'splittable'} (coverage "
            f"{manifest.get('coverage_selected_roots')}, root span share "
            f"{manifest.get('root_span_share')})",
        ),
    ]


@functools.lru_cache(maxsize=None)
def check_bench(bench_dir):
    """Every profile check for one benchmark dir, keyed by check name."""
    torch_trace, config_text = resolve_torch_trace(bench_dir)
    checks = check_trace(
        torch_trace,
        detect_framework(config_text, torch_trace),
        detect_speculative(config_text),
        expected_steps(config_text),
    )
    checks += check_split(torch_trace.parent)
    return {check.name: check for check in checks}
