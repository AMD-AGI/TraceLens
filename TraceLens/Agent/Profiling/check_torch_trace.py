#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Check a Magpie torch_trace directory against TraceLens collection requirements.

Installed with the package. After ``pip install``:

    python -m TraceLens.Agent.Profiling.check_torch_trace <benchmark-or-torch_trace-dir>

Pass a benchmark directory or its ``torch_trace`` directory. The checker
looks at rank 0 only. Requirements are listed in
``TraceLens/Agent/Profiling/trace_collection_requirements.md``.

Checks:

1. Graph-capture traces exist. When the run uses speculative decoding,
   drafter capture traces exist as well.
2. The rank-0 trace carries the detailed user annotations the patches add
   (``sqsq`` / ``sqsk`` aggregates parsed by ``TraceLens/TraceUtils``).
3. That trace contains ``--expected-steps`` engine steps (default 256).
4. SGLang traces contain kernel-shape-tool ops (``sglang_profiler::``) with
   input shapes. Other frameworks skip this check.
5. CPU call stacks are on (``python_function`` events).
6. ``cpu_op`` events carry input shapes (``Input Dims``).

Speculative decoding is read from ``config.yaml`` next to ``torch_trace``
(``--speculative-algorithm``, ``--spec-method``, ``--method eagle``,
``--draft-model``, ``--num-speculative-tokens``). ``SPEC_NUM_TOKENS_OVERRIDE: 0``
turns it off. ``--speculative`` / ``--no-speculative`` override the file.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter
from pathlib import Path

from TraceLens.TraceUtils.utils.annotation_utils import (
    ATOM_DETAILED_PATTERN,
    ATOM_NATIVE_PATTERN,
    SGLANG_DETAILED_PATTERN,
    SGLANG_NATIVE_PATTERN,
    VLLM_DETAILED_PATTERN,
    VLLM_NATIVE_PATTERN,
)

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
INPUT_DIMS_MARK = b'"Input Dims"'
NAME_RE = re.compile(rb'"name": "([^"]*)"')
SHAPE_DIGIT_RE = re.compile(rb"\d")


class Check:
    def __init__(self, name, status, detail):
        self.name = name
        self.status = status
        self.detail = detail

    @property
    def failed(self):
        return self.status == "FAIL"


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
        position = 0
        while True:
            found = data.find(PYTHON_FUNCTION_MARK, position)
            if found < 0 or found >= limit:
                break
            stats["python_function"] += 1
            position = found + len(PYTHON_FUNCTION_MARK)
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
        ):
            checks.append(Check(name, "FAIL", "no single rank-0 trace to scan"))
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
    return checks


def _format_report(label, framework, speculative, checks):
    spec = {True: "on", False: "off", None: "unknown"}[speculative]
    lines = [f"{label}  framework={framework}  speculative={spec}"]
    for check in checks:
        lines.append(f"  {check.status:<4}  {check.name:<24}  {check.detail}")
    failed = [check.name for check in checks if check.failed]
    lines.append("  FAIL  " + ", ".join(failed) if failed else "  PASS")
    return "\n".join(lines), not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="benchmark directory or its torch_trace directory",
    )
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=EXPECTED_STEPS,
        help=f"engine steps required in the rank-0 trace (default {EXPECTED_STEPS})",
    )
    spec = parser.add_mutually_exclusive_group()
    spec.add_argument(
        "--speculative",
        action="store_true",
        help="require drafter capture traces even if config.yaml does not say so",
    )
    spec.add_argument(
        "--no-speculative",
        action="store_true",
        help="do not require drafter capture traces",
    )
    parser.add_argument(
        "--framework",
        choices=("vllm", "sglang", "atom"),
        help="override the framework read from config.yaml",
    )
    args = parser.parse_args(argv)

    all_ok = True
    reports = []
    for raw_path in args.paths:
        torch_trace, config_text = resolve_torch_trace(raw_path)
        framework = args.framework or detect_framework(config_text, torch_trace)
        if args.speculative:
            speculative = True
        elif args.no_speculative:
            speculative = False
        else:
            speculative = detect_speculative(config_text)
        checks = check_trace(
            torch_trace,
            framework,
            speculative,
            args.expected_steps,
        )
        label = torch_trace.parent.name
        report, ok = _format_report(label, framework, speculative, checks)
        reports.append(report)
        all_ok = all_ok and ok
    print("\n\n".join(reports))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
