#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Trace origin/type detection for comparative-analysis routing.

Decides, from a cheap one-pass scan of a Chrome/PyTorch trace, whether a
comparative run should use the structural TraceDiff path or the semantic
comparison path.

Two layers:
  * ``classify_trace(trace_path)`` -> {framework, exec_mode, annotated}
    determines a single trace's origin/type.
  * ``classify_comparison_method(trace1, trace2, capture1_available,
    capture2_available, platform1, platform2)``
    -> {method, reasons, trace1_kind, trace2_kind}
    applies the semantic-vs-tracediff rule across the two traces.

The routing principle is "semantic only when TraceDiff cannot work": the
structural TraceDiff path needs an alignable CPU-op tree on both sides, so
the semantic path is used only when (a) the two traces come from different
frameworks (their CPU-op trees do not align), or (b) a trace is graph-mode
with no capture trace available (no CPU-op tree to diff against). Everything
else -- including same-framework cross-platform eager traces -- stays on
TraceDiff.

This intentionally avoids the heavyweight ``split_vllm_trace()`` (which spawns
a subprocess and fully splits the trace); detection only needs a lightweight
scan of ``traceEvents``.

Usage:
    python comparison_routing.py <trace1> <trace2> \
        --platform1 MI300X --platform2 MI355X \
        --capture1-available --capture2-available
"""

import argparse
import gzip
import json
import re
import sys

# Iteration-annotation markers. Mirrors
# ``TraceLens.TraceUtils.split_inference_trace_annotation.ANNOTATION_PATTERN``
# so detection stays consistent with the trace splitter. These match the
# per-iteration ``user_annotation`` events emitted by vLLM/SGLang.
ANNOTATION_PATTERN = [
    re.compile(
        r"execute_\d+_context_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)_generation_\d+\(sq\d+sk\d+sqsq\d+sqsk\d+\)"
    ),
    re.compile(r"execute_context_\d+\(\d+\)_generation_\d+\(\d+\)"),
    re.compile(r"execute_new_\d+_cached_\d+"),
    re.compile(r"execute_context_\d+\(\d+_\d+\)_generation_\d+\(\d+\)"),
]


def _load_events(trace_path):
    """Load ``traceEvents`` from a .json or .json.gz Chrome trace."""
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", [])


def classify_trace(trace_path):
    """Determine a single trace's origin/type from a cheap one-pass scan.

    Returns:
        dict with
          - ``framework``: "vllm" | "sglang" | "plain"
          - ``exec_mode``: "graph" | "eager"
          - ``annotated``: bool (an inference-framework trace, i.e. carries
            vLLM/SGLang annotations / namespaces)
    """
    events = _load_events(trace_path)

    has_graph_launch = False
    has_vllm = False
    has_sglang = False
    has_annotation = False

    for e in events:
        if not isinstance(e, dict):
            continue
        name = e.get("name", "")
        if not isinstance(name, str):
            continue
        cat = e.get("cat", "")

        # exec_mode: same substring test as extract_trace_data.detect_graph_mode
        # (covers both cudaGraphLaunch and hipGraphLaunch).
        if not has_graph_launch and cat == "cuda_runtime" and "GraphLaunch" in name:
            has_graph_launch = True

        # Framework markers appear in python_function / user_annotation names
        # (module namespaces and file paths).
        if not has_sglang and "sglang" in name:
            has_sglang = True
        if not has_vllm and "vllm" in name:
            has_vllm = True

        # Per-iteration annotations indicate an inference workload.
        if not has_annotation and cat == "user_annotation":
            if any(p.match(name) for p in ANNOTATION_PATTERN):
                has_annotation = True

    if has_sglang:
        framework = "sglang"
    elif has_vllm:
        framework = "vllm"
    else:
        framework = "plain"

    annotated = has_annotation or framework in ("vllm", "sglang")

    return {
        "framework": framework,
        "exec_mode": "graph" if has_graph_launch else "eager",
        "annotated": annotated,
    }


def classify_comparison_method(
    trace1,
    trace2,
    capture1_available=False,
    capture2_available=False,
    platform1=None,
    platform2=None,
):
    """Decide between the structural TraceDiff path and the semantic path.

    Routes to ``semantic`` only when TraceDiff cannot work, i.e. when ANY of:
      - the two traces come from different frameworks (their CPU-op trees do
        not align, so TraceDiff cannot match them), or
      - a trace is graph-mode AND has no capture trace available (no CPU-op
        tree to diff against).

    Everything else stays on ``tracediff``. In particular, a platform
    difference does NOT force semantic: same-framework cross-platform eager
    traces keep an alignable CPU-op tree (only kernel names/durations differ),
    which TraceDiff handles. ``platform1``/``platform2`` are accepted only for
    human-readable reasons, never for the decision.

    Args:
        capture1_available / capture2_available: whether a capture trace was
            collected for the corresponding trace (needed to give a graph-mode
            trace an alignable CPU-op tree). Capture availability is external
            to the trace JSON, so it must be supplied by the caller.

    Returns:
        dict with ``method`` ("semantic"|"tracediff"), a human-readable
        ``reasons`` list, and the per-trace ``classify_trace`` results.
    """
    k1 = classify_trace(trace1)
    k2 = classify_trace(trace2)

    reasons = []
    if k1["framework"] != k2["framework"]:
        reasons.append(
            f"frameworks differ ({k1['framework']} vs {k2['framework']})"
        )
    if k1["exec_mode"] == "graph" and not capture1_available:
        reasons.append("trace1 is graph-mode with no capture trace available")
    if k2["exec_mode"] == "graph" and not capture2_available:
        reasons.append("trace2 is graph-mode with no capture trace available")

    method = "semantic" if reasons else "tracediff"

    return {
        "method": method,
        "reasons": reasons,
        "trace1_kind": k1,
        "trace2_kind": k2,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify two traces and decide TraceDiff vs semantic comparison. "
            "Prints the decision as JSON."
        )
    )
    parser.add_argument("trace1", help="Path to primary trace (.json/.json.gz)")
    parser.add_argument("trace2", help="Path to comparison trace (.json/.json.gz)")
    parser.add_argument(
        "--platform1",
        default=None,
        help="Platform of trace1 (informational only; not used for the decision)",
    )
    parser.add_argument(
        "--platform2",
        default=None,
        help="Platform of trace2 (informational only; not used for the decision)",
    )
    parser.add_argument(
        "--capture1-available",
        action="store_true",
        help="A capture trace was collected for trace1 (gives a graph-mode "
        "trace an alignable CPU-op tree for TraceDiff)",
    )
    parser.add_argument(
        "--capture2-available",
        action="store_true",
        help="A capture trace was collected for trace2",
    )
    args = parser.parse_args()

    result = classify_comparison_method(
        args.trace1,
        args.trace2,
        capture1_available=args.capture1_available,
        capture2_available=args.capture2_available,
        platform1=args.platform1,
        platform2=args.platform2,
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
